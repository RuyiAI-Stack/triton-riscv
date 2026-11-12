// Host regression for NH/E6 trace transport, including persistent mailboxes.
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define NH_E6_MAILBOX_REG ((uintptr_t)0x50000160)
#define NH_E6_RESULT_REG ((uintptr_t)0x50000140)
static const char *const vit_op_trace_messages[] = {"op1 BEGIN\n", "op1 END\n",
                                                    "op2 BEGIN\n", "op2 END\n"};
static uint32_t test_read32(uintptr_t address);
static void test_write32(uintptr_t address, uint32_t value);
static void test_print(const char *text);
static unsigned fences;
// Unaligned neighbouring result cells exercise the control words' complete
// cache-line footprint. Production addresses also require an ELF layout check.
static volatile uint32_t result_before_trace;
#define VIT_OP_TRACE_READ32(address) test_read32(address)
#define VIT_OP_TRACE_WRITE32(address, value) test_write32(address, value)
#define VIT_OP_TRACE_FENCE() (++fences)
#define VIT_OP_TRACE_PRINT(text) test_print(text)
#include "model_op_trace.h"

static volatile uint64_t result_after_trace;
_Static_assert(sizeof(struct vit_op_trace_control) == 64,
               "trace control must occupy its complete cache line");
_Static_assert(_Alignof(struct vit_op_trace_control) == 64,
               "trace control must start on a cache-line boundary");
_Static_assert(offsetof(struct vit_op_trace_control, value) == 0,
               "trace control word must start the cache line");
_Static_assert(_Generic(&vit_op_trace_e6,
                   volatile struct vit_op_trace_control *: 1,
                   default: 0),
               "E6 trace control must remain volatile");
_Static_assert(_Generic(&vit_op_trace_nh,
                   volatile struct vit_op_trace_control *: 1,
                   default: 0),
               "NH trace control must remain volatile");

static uint32_t mailbox, result;
static unsigned nh_active, delayed_polls, polls, prints, acknowledgements;
static char output[256];

static uintptr_t cache_line(const volatile void *address) {
  return (uintptr_t)address & ~(uintptr_t)63;
}

static void check_publication_layout(void) {
  uintptr_t e6 = (uintptr_t)&vit_op_trace_e6;
  uintptr_t nh = (uintptr_t)&vit_op_trace_nh;
  assert((e6 & 63) == 0 && (nh & 63) == 0);
  assert(cache_line(&vit_op_trace_e6) != cache_line(&vit_op_trace_nh));
  const volatile void *results[] = {&result_before_trace, &result_after_trace};
  for (unsigned i = 0; i < sizeof(results) / sizeof(results[0]); ++i) {
    assert(cache_line(results[i]) != cache_line(&vit_op_trace_e6));
    assert(cache_line(results[i]) != cache_line(&vit_op_trace_nh));
  }
  // Trace writes must not disturb adjacent final result storage or padding.
  assert(result_before_trace == UINT32_C(0x12345678));
  assert(result_after_trace == UINT64_C(0x123456789abcdef0));
  for (unsigned i = 0; i < 15; ++i) {
    assert(vit_op_trace_e6.reserved[i] == 0);
    assert(vit_op_trace_nh.reserved[i] == 0);
  }
}

static int consume_on_nh(uint32_t request) {
  assert(!nh_active);
  nh_active = 1;
  int consumed = vit_op_trace_consume(request);
  nh_active = 0;
  return consumed;
}

static uint32_t test_read32(uintptr_t address) {
  // E6 must wait on the NH-owned RESULT, not read/clear its own MAILBOX.
  assert(address == NH_E6_RESULT_REG && !nh_active);
  ++polls;
  if (delayed_polls) {
    --delayed_polls;
    return result; // Includes the previous event's stale ACK.
  }
  assert(consume_on_nh(mailbox));
  assert(consume_on_nh(mailbox));
  assert(consume_on_nh(mailbox));
  return result;
}

static void test_write32(uintptr_t address, uint32_t value) {
  assert(fences);
  if (address == NH_E6_MAILBOX_REG) {
    assert(!nh_active);
    mailbox = value;
  } else {
    assert(address == NH_E6_RESULT_REG && nh_active);
    result = value;
    ++acknowledgements;
    // ACK never clears MAILBOX: this models the failing board observation.
  }
}

static void test_print(const char *text) {
  assert(nh_active);
  assert(strlen(output) + strlen(text) < sizeof(output));
  strcat(output, text);
  ++prints;
}

static void send(uint32_t event) {
  unsigned previous_prints = prints;
  unsigned previous_polls = polls;
  delayed_polls = 3;
  vit_model_trace_event(event);
  assert(polls - previous_polls == 4);   // No stale ACK releases E6 early.
  assert(prints == previous_prints + 1); // Three NH polls print only once.
  assert(result == vit_op_trace_ack(mailbox));
  assert((mailbox & VIT_OP_TRACE_TAG_MASK) == VIT_OP_TRACE_REQUEST);
  assert((mailbox & VIT_OP_TRACE_EVENT_MASK) == event);
}

int main(void) {
  result_before_trace = UINT32_C(0x12345678);
  result_after_trace = UINT64_C(0x123456789abcdef0);
  check_publication_layout();
  assert(!consume_on_nh(0));
  assert(!consume_on_nh(0x1001));
  assert(!prints && !acknowledgements);
  send(0);
  uint32_t first = mailbox;
  send(0); // The same event in a later invocation is still printed.
  assert(first != mailbox);
  send(1);
  send(2);
  send(3);
  assert(prints == 5 && acknowledgements == 15);
  assert(strcmp(output,
                "op1 BEGIN\nop1 BEGIN\nop1 END\nop2 BEGIN\nop2 END\n") == 0);
  assert(consume_on_nh(mailbox));
  assert(prints == 5 && acknowledgements == 16);
  // A last debug ACK must never satisfy the platform's final verdict wait.
  assert(vit_op_trace_completion_result(result) == 0);
  assert(vit_op_trace_completion_result(0) == 0);
  assert(vit_op_trace_completion_result(0xff) == 0xff);
  assert(vit_op_trace_completion_result(0xee) == 0xee);
  check_publication_layout();
  puts("operator trace: cache-line publication, persistent request, stale ACK, "
       "dedup and final verdict PASS");
  return 0;
}
