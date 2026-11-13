// Included once at the end of a model's board_support.c. E6 owns snapshots
// and publication; only NH calls export after observing successful E6 DONE.
// nh_e6 cache helpers, NH_E6_DONE_REG/PASS and print_uart* are provided by the
// including translation unit. The host protocol fixture overrides the fence.
//
#include "scheduler_output_runtime.h"

#ifndef VIT_BOARD_OUTPUT_FENCE
#define VIT_BOARD_OUTPUT_FENCE() \
  __asm__ volatile("fence rw, rw" ::: "memory")
#endif

_Static_assert(VIT_BOARD_EXPECTED_OUTPUTS > 0 && VIT_BOARD_EXPECTED_OUTPUTS <= 16,
               "scheduler output protocol requires 1..16 outputs");
_Static_assert(VIT_BOARD_EXPORT_BYTES > 0 && !(VIT_BOARD_EXPORT_BYTES % 64),
               "snapshot arena must have a positive, 64-byte aligned size");

static const VitBoardOutputMetadata vit_board_output_table[] = VIT_BOARD_OUTPUT_TABLE;
_Static_assert(sizeof(vit_board_output_table) / sizeof(vit_board_output_table[0]) ==
                   VIT_BOARD_EXPECTED_OUTPUTS,
               "output metadata count differs from the generated expectation");

static unsigned char vit_board_output_snapshots[VIT_BOARD_EXPORT_BYTES]
    __attribute__((section(".dma_exports"), aligned(64), used));

typedef struct __attribute__((aligned(64))) {
  volatile unsigned captured[VIT_BOARD_EXPECTED_OUTPUTS];
  volatile unsigned finished;
} VitBoardOutputState;
static VitBoardOutputState vit_board_output_state;

static int vit_board_output_same_string(const char *left, const char *right) {
  if (!left || !right) return 0;
  while (*left && *left == *right) { ++left; ++right; }
  return *left == *right;
}

static int vit_board_output_name_valid(const char *name) {
  if (!name || !((*name >= 'A' && *name <= 'Z') ||
                 (*name >= 'a' && *name <= 'z'))) return 0;
  for (unsigned i = 0; i <= 64; ++i) {
    unsigned char c = (unsigned char)name[i];
    if (!c) return i <= 64;
    if (i == 64 || !((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                    (c >= '0' && c <= '9') || c == '_' || c == '-')) return 0;
  }
  return 0;
}

static int vit_board_output_metadata_valid(void) {
  for (unsigned i = 0; i < VIT_BOARD_EXPECTED_OUTPUTS; ++i) {
    const VitBoardOutputMetadata *entry = &vit_board_output_table[i];
    size_t element_bytes = vit_board_output_same_string(entry->dtype, "|i1") ? 1 :
        (vit_board_output_same_string(entry->dtype, "<f4") ||
         vit_board_output_same_string(entry->dtype, "<i4")) ? 4 : 0;
    if (!vit_board_output_name_valid(entry->name) || !element_bytes ||
        !entry->rank || entry->rank > 4 || !entry->nbytes ||
        entry->offset % 64 || entry->offset > VIT_BOARD_EXPORT_BYTES ||
        entry->nbytes > VIT_BOARD_EXPORT_BYTES - entry->offset) return 0;
    size_t bytes = element_bytes;
    for (unsigned d = 0; d < entry->rank; ++d) {
      if (!entry->shape[d] || entry->shape[d] > SIZE_MAX / bytes) return 0;
      bytes *= (size_t)entry->shape[d];
    }
    if (bytes != entry->nbytes) return 0;
    for (unsigned j = 0; j < i; ++j) {
      const VitBoardOutputMetadata *previous = &vit_board_output_table[j];
      if (vit_board_output_same_string(entry->name, previous->name) ||
          !(entry->offset + entry->nbytes <= previous->offset ||
            previous->offset + previous->nbytes <= entry->offset)) return 0;
    }
  }
  return 1;
}

size_t vit_board_output_bytes(unsigned slot) {
  if (slot >= VIT_BOARD_EXPECTED_OUTPUTS || !vit_board_output_metadata_valid())
    __builtin_trap();
  return vit_board_output_table[slot].nbytes;
}

void vit_board_capture_output(unsigned slot, const void *data, size_t bytes) {
  nh_e6_cache_writes();
  if (slot >= VIT_BOARD_EXPECTED_OUTPUTS || !vit_board_output_metadata_valid() ||
      vit_board_output_state.finished || vit_board_output_state.captured[slot] ||
      !data || bytes != vit_board_output_table[slot].nbytes ||
      (uintptr_t)data > UINTPTR_MAX - bytes) __builtin_trap();
  uintptr_t source_begin = (uintptr_t)data;
  uintptr_t snapshots_begin = (uintptr_t)vit_board_output_snapshots;
  if (!(source_begin + bytes <= snapshots_begin ||
        snapshots_begin + sizeof(vit_board_output_snapshots) <= source_begin))
    __builtin_trap();
  const volatile unsigned char *source = data;
  volatile unsigned char *destination = vit_board_output_snapshots +
      vit_board_output_table[slot].offset;
  // Keep the staged bytes on the stable E6 stack while changing DDR policy.
  // A direct memcpy in either mode can read stale DMA data or retain dirty
  // destination lines. Volatile byte accesses also forbid a compiler memcpy.
  //
  volatile unsigned char chunk[256] __attribute__((aligned(64)));
  for (size_t offset = 0; offset < bytes;) {
    size_t count = bytes - offset;
    if (count > sizeof(chunk)) count = sizeof(chunk);
    nh_e6_cache_uncached_loads();
    for (size_t i = 0; i < count; ++i) chunk[i] = source[offset + i];
    VIT_BOARD_OUTPUT_FENCE();
    nh_e6_cache_writes();
    for (size_t i = 0; i < count; ++i) destination[offset + i] = chunk[i];
    VIT_BOARD_OUTPUT_FENCE();
    offset += count;
  }
  vit_board_output_state.captured[slot] = 1;
  VIT_BOARD_OUTPUT_FENCE();
}

int vit_board_finish_outputs(void) {
  nh_e6_cache_writes();
  if (!vit_board_output_metadata_valid()) return -1;
  for (unsigned slot = 0; slot < VIT_BOARD_EXPECTED_OUTPUTS; ++slot)
    if (!vit_board_output_state.captured[slot]) return -1;
  VIT_BOARD_OUTPUT_FENCE();
  vit_board_output_state.finished = 1;
  VIT_BOARD_OUTPUT_FENCE();
  return 0;
}

unsigned vit_board_outputs_ready(void) {
  VIT_BOARD_OUTPUT_FENCE();
  return vit_board_output_state.finished == 1;
}

// uart.h has only dec32; shapes and lengths are uint64_t in the protocol.
static void vit_board_output_decimal(uint64_t value) {
  char digits[21];
  unsigned end = sizeof(digits) - 1;
  digits[end] = 0;
  do { digits[--end] = (char)('0' + value % 10); value /= 10; } while (value);
  print_uart(digits + end);
}

int vit_board_export_outputs(void) {
  if (*(volatile uint32_t *)NH_E6_DONE_REG != NH_E6_DONE_PASS ||
      !vit_board_outputs_ready() || !vit_board_output_metadata_valid()) return -1;
  // The resident DDR capture path accepts a 32-bit physical-address string,
  // even though the generic UART protocol parser can represent uint64_t.
  // Reject the full range before printing any declarations.
  //
  for (unsigned slot = 0; slot < VIT_BOARD_EXPECTED_OUTPUTS; ++slot) {
    const VitBoardOutputMetadata *entry = &vit_board_output_table[slot];
    uintptr_t address = (uintptr_t)(vit_board_output_snapshots + entry->offset);
    if (address > UINT32_MAX || entry->nbytes - 1 > UINT32_MAX - address)
      return -1;
  }
  // Validate every record before emitting the first line. Values/names are
  // constrained above, so no JSON escaping or partial invalid record is needed.
  //
  for (unsigned slot = 0; slot < VIT_BOARD_EXPECTED_OUTPUTS; ++slot) {
    const VitBoardOutputMetadata *entry = &vit_board_output_table[slot];
    print_uart("@@FPGA_OUTPUT {\"version\":1,\"name\":\"");
    print_uart(entry->name);
    print_uart("\",\"address\":\"0x");
    print_uart_hex32((uint32_t)(uintptr_t)(vit_board_output_snapshots + entry->offset));
    print_uart("\",\"nbytes\":"); vit_board_output_decimal(entry->nbytes);
    print_uart(",\"dtype\":\""); print_uart(entry->dtype);
    print_uart("\",\"shape\":[");
    for (unsigned d = 0; d < entry->rank; ++d) {
      if (d) print_uart(",");
      vit_board_output_decimal(entry->shape[d]);
    }
    print_uart("]}\r\n");
  }
  return 0;
}
