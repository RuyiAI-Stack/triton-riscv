//
// model_op_trace.h for BOSCAME FPGA model 'vit-base-patch16-224'.
//
// Inputs: buffers, headers, or board state owned by this model tree.
//
// Outputs: the symbols the case harness or model runtime expects.
//
// When used: built with the model makefile under
// examples/BOSCAMEFPGA/vit-base-patch16-224.
//

#ifndef VIT_MODEL_OP_TRACE_H
#define VIT_MODEL_OP_TRACE_H

#include <stdint.h>

// E6 owns MAILBOX requests; NH owns RESULT acknowledgements. Never require
// a write by NH to clear the E6 request register. All message text is immutable
// and selected by event number, so the protocol shares no mutable DDR pointer.
// The including translation unit supplies vit_op_trace_messages[].
#define VIT_OP_TRACE_TAG_MASK UINT32_C(0xf0000000)
#define VIT_OP_TRACE_REQUEST UINT32_C(0xe0000000)
#define VIT_OP_TRACE_ACK UINT32_C(0xf0000000)
#define VIT_OP_TRACE_EVENT_MASK UINT32_C(0xfff)
#define VIT_OP_TRACE_SEQUENCE_MAX UINT32_C(0xffff)

#ifndef VIT_OP_TRACE_READ32
#define VIT_OP_TRACE_READ32(address) (*(volatile uint32_t *)(address))
#endif
#ifndef VIT_OP_TRACE_WRITE32
#define VIT_OP_TRACE_WRITE32(address, value)                                   \
  (*(volatile uint32_t *)(address) = (value))
#endif
#ifndef VIT_OP_TRACE_FENCE
#define VIT_OP_TRACE_FENCE() __asm__ volatile("fence iorw, iorw" ::: "memory")
#endif
#ifndef VIT_OP_TRACE_PRINT
#define VIT_OP_TRACE_PRINT(text) print_uart(text)
#endif

// The cores do not share coherent caches. In particular, NH repeatedly reads
// and writes last_request while E6 publishes its final result cells later.
// Keep each writer's control word on its own cache line so NH cannot cache
// adjacent E6 result cells before their publication.
struct vit_op_trace_control {
  uint32_t value;
  uint32_t reserved[15];
} __attribute__((aligned(64)));
static volatile struct vit_op_trace_control vit_op_trace_e6;
static volatile struct vit_op_trace_control vit_op_trace_nh;
#define vit_op_trace_sequence vit_op_trace_e6.value
#define vit_op_trace_last_request vit_op_trace_nh.value

static uint32_t vit_op_trace_ack(uint32_t request) {
  return VIT_OP_TRACE_ACK | (request & ~VIT_OP_TRACE_TAG_MASK);
}

void vit_model_trace_event(uint32_t event) {
  if (event >=
          sizeof(vit_op_trace_messages) / sizeof(vit_op_trace_messages[0]) ||
      event > VIT_OP_TRACE_EVENT_MASK ||
      vit_op_trace_sequence == VIT_OP_TRACE_SEQUENCE_MAX)
    __builtin_trap();
  uint32_t request =
      VIT_OP_TRACE_REQUEST | (++vit_op_trace_sequence << 12) | event;
  VIT_OP_TRACE_FENCE();
  VIT_OP_TRACE_WRITE32(NH_E6_MAILBOX_REG, request);
  VIT_OP_TRACE_FENCE();
  while (VIT_OP_TRACE_READ32(NH_E6_RESULT_REG) != vit_op_trace_ack(request))
    VIT_OP_TRACE_FENCE();
}

// NH may observe a persistent request many times. Print it once, but always
// repeat its acknowledgement. E6 cannot post the next event until this ACK.
static int vit_op_trace_consume(uint32_t request) {
  if ((request & VIT_OP_TRACE_TAG_MASK) != VIT_OP_TRACE_REQUEST)
    return 0;
  uint32_t event = request & VIT_OP_TRACE_EVENT_MASK;
  if (!(request & UINT32_C(0x0ffff000)) ||
      event >= sizeof(vit_op_trace_messages) / sizeof(vit_op_trace_messages[0]))
    __builtin_trap();
  if (request != vit_op_trace_last_request) {
    VIT_OP_TRACE_PRINT(vit_op_trace_messages[event]);
    vit_op_trace_last_request = request;
  }
  VIT_OP_TRACE_FENCE();
  VIT_OP_TRACE_WRITE32(NH_E6_RESULT_REG, vit_op_trace_ack(request));
  VIT_OP_TRACE_FENCE();
  return 1;
}

// The platform's final E6 completion loop accepts any nonzero RESULT. Hide
// debug ACKs there; the event handshake above uses raw MMIO loads.
static uint32_t vit_op_trace_completion_result(uint32_t value) {
  return (value & VIT_OP_TRACE_TAG_MASK) == VIT_OP_TRACE_ACK ? 0u : value;
}

#endif // VIT_MODEL_OP_TRACE_H
