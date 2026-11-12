// Whole-model ViT-B/16 E6 support layer.
//
// Provides the C runtime the lowered graph and the generated launcher need
// (bump allocator, memrefCopy, memcpy/memset, UART/report hooks) and the two
// NH/E6 application hooks the platform expects.  Numerical work lives in the
// separately compiled BOSCAME kernels; nothing here computes model values.
//
#include <stddef.h>
#include <stdint.h>

#include "bosc_ame_dma.h"
#include "nh_e6_platform.h"
#include "uart.h"

#include "model_runtime.h"

// ------------------------------------------------------------------
// Result cells (BSS: cleared by e6_start, written by e6_main)
// ------------------------------------------------------------------

// NH must first read these cells after DONE. Isolate the result block from
// the trace controls NH accesses throughout execution.
volatile unsigned model_returned __attribute__((aligned(64)));
volatile unsigned model_validation_executed;
volatile unsigned model_checked;
volatile unsigned model_errors;
volatile uint64_t model_cycles;
volatile uint64_t model_logits_fnv64;
volatile uint32_t model_top1;
volatile uint32_t model_max_abs_bits;

// Trap reporting/context cells (BSS: set by the trap entry or the launcher).
volatile uint64_t vit_kernel_case;  // 1-based typed-kernel ordinal in flight
volatile uint64_t vit_kernel_calls; // number of wrapped kernel calls started
volatile uint64_t vit_trap_sp;
volatile uint64_t vit_trap_case;
volatile uint64_t vit_trap_calls;

extern void vit_e6_trap_entry(void);

// The stock platform trap entry publishes the diagnostic cells without
// switching CSR 0xbc0 back to the write-through store policy.  After a kernel
// (DMA/private-DDR) phase the uncached-load policy is still active, so those
// stores can stay in the E6 cache while NH reads the DDR copy: the reported
// mcause/mepc/mtval become zeros.  Install an entry that forces the policy
// first and records the in-flight kernel call.
void vit_install_trap_handler(void) {
  __asm__ volatile("csrw mtvec, %0" ::"r"((uintptr_t)vit_e6_trap_entry)
                   : "memory");
}

// ------------------------------------------------------------------
// Bump allocator for the lowered graph's transient allocations
// ------------------------------------------------------------------

extern char __model_heap_start[] __attribute__((weak));
extern char __model_heap_end[] __attribute__((weak));

static uintptr_t model_heap_top;

static void *model_heap_alloc(size_t bytes) {
  if (!__model_heap_start || !__model_heap_end)
    __builtin_trap();
  if (model_heap_top == 0u)
    model_heap_top = (uintptr_t)__model_heap_start;
  uintptr_t base = (model_heap_top + 63u) & ~(uintptr_t)63u;
  uintptr_t end = base + bytes;
  if (end < base || end > (uintptr_t)__model_heap_end)
    __builtin_trap();
  model_heap_top = end;
  return (void *)base;
}

void *malloc(size_t bytes) { return model_heap_alloc(bytes); }
void free(void *pointer) { (void)pointer; }
void *calloc(size_t count, size_t size) {
  size_t bytes = count * size;
  void *pointer = model_heap_alloc(bytes);
  unsigned char *out = pointer;
  for (size_t i = 0; i < bytes; ++i)
    out[i] = 0;
  return pointer;
}
void *realloc(void *pointer, size_t bytes) {
  (void)pointer;
  return model_heap_alloc(bytes);
}

// ------------------------------------------------------------------
// memrefCopy (MLIR CRunnerUtils semantics, unranked descriptors)
// ------------------------------------------------------------------

typedef struct {
  int64_t rank;
  void *descriptor;
} ModelUnrankedMemRef;

struct ModelMemRefDescriptor {
  void *allocated;
  void *aligned;
  int64_t offset;
  // int64_t sizes[rank]; int64_t strides[rank]; follow.
};

static void model_copy_strided(int64_t rank, int64_t elem_size, int64_t dim,
                               char *src, const int64_t *src_strides, char *dst,
                               const int64_t *dst_strides,
                               const int64_t *sizes) {
  if (dim == rank) {
    for (int64_t i = 0; i < elem_size; ++i)
      dst[i] = src[i];
    return;
  }
  for (int64_t i = 0; i < sizes[dim]; ++i)
    model_copy_strided(rank, elem_size, dim + 1,
                       src + i * src_strides[dim] * elem_size, src_strides,
                       dst + i * dst_strides[dim] * elem_size, dst_strides,
                       sizes);
}

void memrefCopy(int64_t elem_size, ModelUnrankedMemRef *src_arg,
                ModelUnrankedMemRef *dst_arg) {
  if (!src_arg || !dst_arg || src_arg->rank != dst_arg->rank)
    __builtin_trap();
  int64_t rank = src_arg->rank;
  struct ModelMemRefDescriptor *src =
      (struct ModelMemRefDescriptor *)src_arg->descriptor;
  struct ModelMemRefDescriptor *dst =
      (struct ModelMemRefDescriptor *)dst_arg->descriptor;
  int64_t *src_sizes = (int64_t *)(src + 1);
  int64_t *src_strides = src_sizes + rank;
  int64_t *dst_strides = (int64_t *)(dst + 1) + rank;
  char *src_ptr = (char *)src->aligned + src->offset * elem_size;
  char *dst_ptr = (char *)dst->aligned + dst->offset * elem_size;
  model_copy_strided(rank, elem_size, 0, src_ptr, src_strides, dst_ptr,
                     dst_strides, src_sizes);
}

// ------------------------------------------------------------------
// Freestanding C helpers
// ------------------------------------------------------------------

void *memcpy(void *destination, const void *source, size_t count) {
  unsigned char *out = destination;
  const unsigned char *in = source;
  for (size_t i = 0; i < count; ++i)
    out[i] = in[i];
  return destination;
}

void *memmove(void *destination, const void *source, size_t count) {
  unsigned char *out = destination;
  const unsigned char *in = source;
  if (out < in) {
    for (size_t i = 0; i < count; ++i)
      out[i] = in[i];
  } else {
    for (size_t i = count; i != 0; --i)
      out[i - 1] = in[i - 1];
  }
  return destination;
}

void *memset(void *destination, int value, size_t count) {
  unsigned char *out = destination;
  for (size_t i = 0; i < count; ++i)
    out[i] = (unsigned char)value;
  return destination;
}

void abort(void) { __builtin_trap(); }

uint64_t nr_cycles(void) {
  uint64_t value;
  __asm__ volatile("rdcycle %0" : "=r"(value)::"memory");
  return value;
}

#ifdef MODEL_OP_TRACE
#include "model_op_trace.h"
#include "model_op_trace_messages.h"
#endif

void nr_puts(const char *text) { (void)text; }
void nr_hex32(uint32_t value) { (void)value; }
void nr_hex64(uint64_t value) { (void)value; }
void nr_copy_bytes(void *destination, const void *source, size_t bytes) {
  memcpy(destination, source, bytes);
}
float fabsf(float value) { return __builtin_fabsf(value); }
void ame_fence(void) { __asm__ volatile("fence iorw, iorw" ::: "memory"); }

// ------------------------------------------------------------------
// NH/E6 application hooks
// ------------------------------------------------------------------

int e6_main(void); // generated launcher

void board_init_uart(uint32_t frequency, uint32_t baud) {
  init_uart(frequency, baud);
  print_uart("@@FPGA_START {\"version\":1}\r\n");
}

uint32_t vit_board_mmio_read32(uintptr_t address) {
  uint32_t value = *(volatile uint32_t *)address;
#ifdef MODEL_OP_TRACE
  // The NH platform wait loop already uses this model-specific MMIO hook.
  // Consume trace requests here without changing the shared runtime.
  if (address == NH_E6_MAILBOX_REG && vit_op_trace_consume(value))
    return 0u;
  if (address == NH_E6_RESULT_REG)
    return vit_op_trace_completion_result(value);
#endif
  return value;
}

const char *nh_e6_app_tag(void) { return "vit-model"; }

// Report a private-DDR arena failure without trapping.  Each field gets its
// own tag nibble so the payload bits are never masked by an OR (the earlier
// 0xE100-style encoding was ambiguous).  NH prints every new marker; the long
// delay keeps consecutive markers from being coalesced.  Fields: reason, arena
// top (high/low 16), requested bytes (high/low 16), and the runtime
// acquire/release counters plus peak live bytes. Internal runtime trace
// records are printed directly by NH after E6 stops.
static void vit_private_ddr_marker(uint32_t marker) {
  nh_e6_progress(marker);
  __asm__ volatile("" ::: "memory");
  for (volatile uint32_t i = 0; i < 500000u; ++i)
    __asm__ volatile("nop");
}

void vit_publish_private_ddr_failure(uint32_t reason, uint64_t address,
                                     uint64_t bytes) {
  vit_private_ddr_marker(0x10000000u | (reason & 0xFFFFu));
  vit_private_ddr_marker(0x20000000u | (uint32_t)((address >> 16) & 0xFFFFu));
  vit_private_ddr_marker(0x30000000u | (uint32_t)(address & 0xFFFFu));
  vit_private_ddr_marker(0x40000000u | (uint32_t)((bytes >> 16) & 0xFFFFu));
  vit_private_ddr_marker(0x50000000u | (uint32_t)(bytes & 0xFFFFu));
  vit_private_ddr_marker(0x60000000u |
                         (uint32_t)(bosc_ame_private_ddr_acquires & 0xFFFFFFu));
  vit_private_ddr_marker(0x70000000u |
                         (uint32_t)(bosc_ame_private_ddr_releases & 0xFFFFFFu));
  vit_private_ddr_marker(
      0x80000000u | (uint32_t)(bosc_ame_private_ddr_peak_bytes & 0xFFFFFFu));
  __asm__ volatile("fence iorw, iorw" ::: "memory");
  *(volatile uint32_t *)(uintptr_t)NH_E6_DONE_REG = NH_E6_DONE_FAIL;
  __asm__ volatile("fence iorw, iorw" ::: "memory");
  for (;;)
    __asm__ volatile("wfi");
}

#ifdef MODEL_DDR_TRACE
// NH reads the published DDR records after E6 has stopped. No E6-side loads
// or marker delays are needed to reconstruct allocator state transitions.
static void vit_report_private_ddr(uint32_t done) {
  unsigned count = bosc_ame_private_ddr_trace_count;
  nh_e6_print_tag_prefix();
  print_uart("ddr acquires=");
  print_uart_dec32((uint32_t)bosc_ame_private_ddr_acquires);
  print_uart(" releases=");
  print_uart_dec32((uint32_t)bosc_ame_private_ddr_releases);
  print_uart(" peak=");
  print_uart_dec32((uint32_t)bosc_ame_private_ddr_peak_bytes);
  print_uart(" trace_count=");
  print_uart_dec32(count);
  print_uart("\r\n");
  if (done == NH_E6_DONE_PASS)
    return;
  unsigned first = count > BOSC_AME_PRIVATE_DDR_TRACE_EVENTS
                       ? count - BOSC_AME_PRIVATE_DDR_TRACE_EVENTS
                       : 0u;
  for (unsigned i = first; i < count; ++i) {
    const volatile struct bosc_ame_private_ddr_trace_event *event =
        &bosc_ame_private_ddr_trace[i % BOSC_AME_PRIVATE_DDR_TRACE_EVENTS];
    nh_e6_print_tag_prefix();
    print_uart("ddr ");
    print_uart_dec32(i);
    print_uart(" op=0x");
    print_uart_hex32((uint32_t)event->op);
    print_uart(" bytes=");
    print_uart_dec32((uint32_t)event->bytes);
    print_uart(" address=0x");
    print_uart_hex64(event->address);
    print_uart(" top=0x");
    print_uart_hex64(event->top_before);
    print_uart(" active=0x");
    print_uart_hex64(event->active_before);
    print_uart(" after_top=0x");
    print_uart_hex64(event->top_after);
    print_uart(" after_active=0x");
    print_uart_hex64(event->active_after);
    print_uart(" acquires=");
    print_uart_dec32((uint32_t)event->acquires);
    print_uart(" releases=");
    print_uart_dec32((uint32_t)event->releases);
    print_uart("\r\n");
  }
}
#endif

void nh_e6_app_report(void) {
  nh_e6_print_tag_prefix();
  print_uart("kernel_returned=");
  print_uart_dec32((uint32_t)model_returned);
  print_uart(" validation_executed=");
  print_uart_dec32((uint32_t)model_validation_executed);
  print_uart(" checked=");
  print_uart_dec32((uint32_t)model_checked);
  print_uart(" errors=");
  print_uart_dec32((uint32_t)model_errors);
  print_uart(" max_abs_error_f32_bits=0x");
  print_uart_hex32((uint32_t)model_max_abs_bits);
  print_uart(" logits_fnv64=0x");
  print_uart_hex64((uint64_t)model_logits_fnv64);
  print_uart(" top1=");
  print_uart_dec32((uint32_t)model_top1);
  print_uart(" cycles=0x");
  print_uart_hex64((uint64_t)model_cycles);
  print_uart("\r\n");

  uint32_t done = *(volatile uint32_t *)NH_E6_DONE_REG;
#ifdef MODEL_DDR_TRACE
  vit_report_private_ddr(done);
#endif
  if (done == NH_E6_DONE_TRAP) {
    nh_e6_print_tag_prefix();
    print_uart("trap_sp=0x");
    print_uart_hex64((uint64_t)vit_trap_sp);
    print_uart(" trap_case=");
    print_uart_dec32((uint32_t)vit_trap_case);
    print_uart(" trap_calls=");
    print_uart_dec32((uint32_t)vit_trap_calls);
    print_uart("\r\n");
    print_uart(
        "@@FPGA_RESULT {\"version\":1,\"status\":\"trap\",\"mcause\":\"0x");
    print_uart_hex64(e6_diag_mcause);
    print_uart("\",\"mepc\":\"0x");
    print_uart_hex64(e6_diag_mepc);
    print_uart("\",\"mtval\":\"0x");
    print_uart_hex64(e6_diag_mtval);
    print_uart("\"}\r\n");
    return;
  }
  int ok = done == NH_E6_DONE_PASS && model_validation_executed &&
           model_returned == 1u && model_checked == 1000u && model_errors == 0u;
  print_uart(
      ok ? "@@FPGA_RESULT "
           "{\"version\":1,\"status\":\"exited\",\"app_exit_code\":0}\r\n"
         : "@@FPGA_RESULT "
           "{\"version\":1,\"status\":\"exited\",\"app_exit_code\":1}\r\n");
}
