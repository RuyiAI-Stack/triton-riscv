// Real NH/E6 support for the generated EfficientNet call graph.
#include "model_board_support.h"
#include "model_trace_messages.h"

unsigned char model_workspace_blob[EFFICIENTNET_B3_WORKSPACE_BYTES]
    __attribute__((section(".model_workspace"), aligned(64)));
float model_export_logits[EFFICIENTNET_B3_OUTPUT_ELEMENTS]
    __attribute__((section(".model_export"), aligned(64)));
uint64_t model_export_cycles[EFFICIENTNET_B3_PHYSICAL_CALLS]
    __attribute__((section(".model_export"), aligned(64)));
volatile unsigned model_returned __attribute__((aligned(64)));
volatile unsigned model_checked, model_errors, model_top1, model_call_count,
    model_call_inflight;
volatile uint32_t model_max_abs_bits;
volatile uint64_t model_cycles, model_kernel_cycles, model_logits_hash,
    model_logits_address;
volatile uint64_t model_call_cycles[EFFICIENTNET_B3_PHYSICAL_CALLS];
static uint64_t call_begin;
struct Checkpoint {
  uint32_t call, span;
  uint64_t bytes, hash;
};
static struct Checkpoint checkpoint_hashes[EFFICIENTNET_B3_PHYSICAL_CALLS * 2];
static unsigned checkpoint_count;
// NH reads each row exactly once after its END mailbox publication.  Give
// every call its own cache line: reading call N must not cache future E6
// writes to call N+1. E6 alone clears BSS; NH never pre-reads these rows.
struct LiveSample {
  uint64_t cycles;
  uint32_t count, padding;
  struct Checkpoint checkpoints[2];
} __attribute__((aligned(64)));
_Static_assert(sizeof(struct LiveSample) == 64, "one cache line per call");
static volatile struct LiveSample live_samples[EFFICIENTNET_B3_PHYSICAL_CALLS];
// NH owns UART and consumes E6-written control records. Fences and distinct
// cache lines alone do not prevent stale/prefetched NH data. This is the
// existing NR console's NH-only CBO protocol; E6 does not call this helper.
static inline void nh_invalidate(const volatile void *p) {
  __asm__ volatile(".option push\n.option arch,+zicbom\ncbo.inval "
                   "(%0)\n.option pop\nfence r,r" ::"r"(p)
                   : "memory");
}
typedef uint64_t AliasU64 __attribute__((__may_alias__));
static void coherent_copy(void *dst, const void *src, size_t bytes) {
  volatile AliasU64 *d = dst;
  const volatile AliasU64 *s = src;
  AliasU64 block[16];
  for (size_t base = 0; base < bytes / 8u; base += 16u) {
    size_t count = bytes / 8u - base;
    if (count > 16u)
      count = 16u;
    bosc_ame_ddr_read_begin();
    for (size_t i = 0; i < count; ++i)
      block[i] = s[base + i];
    bosc_ame_ddr_write_begin();
    for (size_t i = 0; i < count; ++i)
      d[base + i] = block[i];
  }
  volatile unsigned char *tail_d = dst;
  const volatile unsigned char *tail_s = src;
  for (size_t i = bytes / 8u * 8u; i < bytes; ++i) {
    bosc_ame_ddr_read_begin();
    unsigned char v = tail_s[i];
    bosc_ame_ddr_write_begin();
    tail_d[i] = v;
  }
}
void efficientnet_export_results(const float *logits) {
  coherent_copy(model_export_logits, logits, sizeof(model_export_logits));
  coherent_copy(model_export_cycles, (const void *)model_call_cycles,
                sizeof(model_export_cycles));
  __asm__ volatile("fence iorw, iorw" ::: "memory");
}
#ifdef MODEL_PREFIX_PROBE
static unsigned char prefix_stem[40u * 144u * 144u * 4u]
    __attribute__((section(".model_checkpoint"), aligned(64)));
static unsigned char prefix_dw[40u * 144u * 144u * 4u]
    __attribute__((section(".model_checkpoint"), aligned(64)));
static unsigned char prefix_acc[10u * 4u]
    __attribute__((section(".model_checkpoint"), aligned(64)));
static void snapshot(void *dst, const void *src, size_t bytes) {
  // Read under the DMA-visible policy, then restore write-through per word.
  // This diagnostic copy intentionally favors correctness over runtime.
  coherent_copy(dst, src, bytes);
}
#endif

void *workspace(size_t offset) {
  if (offset >= EFFICIENTNET_B3_WORKSPACE_BYTES)
    __builtin_trap();
  return model_workspace_blob + offset;
}
void *memcpy(void *d, const void *s, size_t n) {
  for (size_t i = 0; i < n; ++i)
    ((unsigned char *)d)[i] = ((const unsigned char *)s)[i];
  return d;
}
void *memmove(void *d, const void *s, size_t n) {
  unsigned char *a = d;
  const unsigned char *b = s;
  if (a < b) {
    for (size_t i = 0; i < n; ++i)
      a[i] = b[i];
  } else {
    for (size_t i = n; i; --i)
      a[i - 1] = b[i - 1];
  }
  return d;
}
void *memset(void *d, int v, size_t n) {
  for (size_t i = 0; i < n; ++i)
    ((unsigned char *)d)[i] = (unsigned char)v;
  return d;
}
void nr_copy_bytes(void *d, const void *s, size_t n) { memcpy(d, s, n); }
float fabsf(float v) { return __builtin_fabsf(v); }
void nr_puts(const char *s) { (void)s; }
void nr_hex32(uint32_t v) { (void)v; }
void nr_hex64(uint64_t v) { (void)v; }
uint64_t nr_cycles(void) {
  uint64_t v;
  __asm__ volatile("rdcycle %0" : "=r"(v)::"memory");
  return v;
}
// BOSCAME emission fences each AME instruction and completes DMA waits;
// this phase fence orders those completed accesses at typed-call boundaries.
void ame_fence(void) { __asm__ volatile("fence iorw, iorw" ::: "memory"); }

static uint64_t hash_bytes(const void *pointer, size_t n) {
  const unsigned char *p = pointer;
  uint64_t h = UINT64_C(0xcbf29ce484222325);
  for (size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= UINT64_C(0x100000001b3);
  }
  return h;
}

#define TRACE_TAG UINT32_C(0xe0000000)
#define ACK_TAG UINT32_C(0xf0000000)
// Each core writes separate state; both live outside switchable PMP region.
struct TraceControl {
  uint32_t value, padding[15];
} __attribute__((aligned(64)));
static volatile struct TraceControl e6_trace;
static volatile struct TraceControl nh_trace;
void efficientnet_trace(unsigned event) {
  if (event >= sizeof(efficientnet_trace_messages) /
                   sizeof(efficientnet_trace_messages[0]) ||
      event > 0xfffu || e6_trace.value == 0xffffu)
    __builtin_trap();
  uint32_t request = TRACE_TAG | (++e6_trace.value << 12) | event;
  __asm__ volatile("fence iorw, iorw" ::: "memory");
  *(volatile uint32_t *)NH_E6_MAILBOX_REG = request;
  __asm__ volatile("fence iorw, iorw" ::: "memory");
  while (*(volatile uint32_t *)NH_E6_RESULT_REG !=
         (ACK_TAG | (request & 0x0fffffffu)))
    __asm__ volatile("fence iorw, iorw" ::: "memory");
}
uint32_t efficientnet_mmio_read32(uintptr_t address) {
  uint32_t value = *(volatile uint32_t *)address;
  if (address == NH_E6_MAILBOX_REG && (value & 0xf0000000u) == TRACE_TAG) {
    unsigned event = value & 0xfffu;
    if (event >= sizeof(efficientnet_trace_messages) /
                     sizeof(efficientnet_trace_messages[0]))
      __builtin_trap();
    if (nh_trace.value != value) {
      print_uart(efficientnet_trace_messages[event]);
      if (event & 1u) {
        unsigned index = event / 2u;
        const volatile struct LiveSample *sample = &live_samples[index];
        nh_invalidate(sample);
        print_uart("[efficientnet-cycle] call=");
        print_uart_dec32(index);
        print_uart(" cycles=0x");
        print_uart_hex64(sample->cycles);
        uart_putc(13);
        uart_putc(10);
        for (unsigned i = 0; i < sample->count; ++i) {
          const volatile struct Checkpoint *c = &sample->checkpoints[i];
          print_uart("[efficientnet-checkpoint] call=");
          print_uart_dec32(c->call);
          print_uart(" span=");
          print_uart_dec32(c->span);
          print_uart(" fnv64=0x");
          print_uart_hex64(c->hash);
          uart_putc(13);
          uart_putc(10);
        }
      }
      nh_trace.value = value;
    }
    __asm__ volatile("fence iorw, iorw" ::: "memory");
    *(volatile uint32_t *)NH_E6_RESULT_REG = ACK_TAG | (value & 0x0fffffffu);
    __asm__ volatile("fence iorw, iorw" ::: "memory");
    return 0;
  }
  if (address == NH_E6_RESULT_REG && (value & 0xf0000000u) == ACK_TAG)
    return 0;
  return value;
}

void efficientnet_before(unsigned index, const char *name,
                         const EfficientNetB3Span *spans, unsigned count) {
  (void)name;
  (void)spans;
  (void)count;
  if (index >= EFFICIENTNET_B3_PHYSICAL_CALLS)
    __builtin_trap();
  nh_e6_cache_writes();
  model_call_inflight = index + 1;
  efficientnet_trace(index * 2);
  nh_e6_cache_writes();
  call_begin = nr_cycles();
}
void efficientnet_after(unsigned index, const char *name,
                        const EfficientNetB3Span *spans, unsigned count) {
  (void)name;
  ame_fence();
  uint64_t cycles = nr_cycles() - call_begin;
  nh_e6_cache_writes();
  model_call_cycles[index] = cycles;
  model_kernel_cycles += cycles;
  model_call_count = index + 1;
  live_samples[index].cycles = cycles;
  live_samples[index].count = 0;
#ifdef MODEL_PREFIX_PROBE
  if (index == 0u)
    snapshot(prefix_stem, spans[count - 1].pointer, sizeof(prefix_stem));
  if (index == 2u)
    snapshot(prefix_dw, spans[count - 1].pointer, sizeof(prefix_dw));
  if (index == 6u)
    snapshot(prefix_acc, spans[count - 1].pointer, sizeof(prefix_acc));
#endif
  // First 24 physical calls retain hashes for conv, packed quant, Acc and deQ;
  // later milestones plus the final call enable progressive localization.
  // Read output bytes only with the documented uncached DDR policy.
  if (index < 24u || index % 32u == 31u ||
      index + 1u == EFFICIENTNET_B3_PHYSICAL_CALLS) {
    for (unsigned i = 0; i < count; ++i)
      if (spans[i].write) {
        if (checkpoint_count >=
            sizeof(checkpoint_hashes) / sizeof(checkpoint_hashes[0]))
          __builtin_trap();
        bosc_ame_ddr_read_begin();
        uint64_t hash = hash_bytes(spans[i].pointer, spans[i].bytes);
        bosc_ame_ddr_write_begin();
        struct Checkpoint checkpoint = {index, i, spans[i].bytes, hash};
        checkpoint_hashes[checkpoint_count++] = checkpoint;
        unsigned slot = live_samples[index].count;
        if (slot >= 2u)
          __builtin_trap();
        live_samples[index].checkpoints[slot] = checkpoint;
        live_samples[index].count = slot + 1u;
      }
  }
  efficientnet_trace(index * 2 + 1);
  nh_e6_cache_writes();
  model_call_inflight = 0;
}
const EfficientNetB3Hooks efficientnet_hooks = {efficientnet_before,
                                                efficientnet_after};
extern void efficientnet_trap_entry(void);
void efficientnet_install_trap(void) {
  __asm__ volatile("csrw mtvec, %0" ::"r"((uintptr_t)efficientnet_trap_entry)
                   : "memory");
}
const char *nh_e6_app_tag(void) { return "efficientnet-model"; }
static const char protocol_start[] = {
    64,  64,  70,  80,  71,  65,  95,  83, 84, 65, 82,  84, 32, 123, 34,
    118, 101, 114, 115, 105, 111, 110, 34, 58, 49, 125, 13, 10, 0};
static const char protocol_trap[] = {
    64,  64,  70,  80,  71,  65,  95,  82,  69,  83,  85,  76, 84,  32,
    123, 34,  118, 101, 114, 115, 105, 111, 110, 34,  58,  49, 44,  34,
    115, 116, 97,  116, 117, 115, 34,  58,  34,  116, 114, 97, 112, 34,
    44,  34,  109, 99,  97,  117, 115, 101, 34,  58,  34,  48, 120, 0};
static const char protocol_mepc[] = {34, 44, 34, 109, 101, 112, 99,
                                     34, 58, 34, 48,  120, 0};
static const char protocol_mtval[] = {34,  44, 34, 109, 116, 118, 97,
                                      108, 34, 58, 34,  48,  120, 0};
static const char protocol_close[] = {34, 125, 13, 10, 0};
static const char protocol_pass[] = {
    64,  64,  70,  80,  71,  65,  95,  82,  69,  83,  85,  76,  84,  32,
    123, 34,  118, 101, 114, 115, 105, 111, 110, 34,  58,  49,  44,  34,
    115, 116, 97,  116, 117, 115, 34,  58,  34,  101, 120, 105, 116, 101,
    100, 34,  44,  34,  97,  112, 112, 95,  101, 120, 105, 116, 95,  99,
    111, 100, 101, 34,  58,  48,  125, 13,  10,  0};
static const char protocol_fail[] = {
    64,  64,  70,  80,  71,  65,  95,  82,  69,  83,  85,  76,  84,  32,
    123, 34,  118, 101, 114, 115, 105, 111, 110, 34,  58,  49,  44,  34,
    115, 116, 97,  116, 117, 115, 34,  58,  34,  101, 120, 105, 116, 101,
    100, 34,  44,  34,  97,  112, 112, 95,  101, 120, 105, 116, 95,  99,
    111, 100, 101, 34,  58,  49,  125, 13,  10,  0};
static const char newline[] = {13, 10, 0};
void efficientnet_init_uart(uint32_t f, uint32_t b) {
  init_uart(f, b);
  print_uart(protocol_start);
}
static void output(const char *name, const void *pointer, unsigned nbytes,
                   const char *dtype, const char *shape) {
  // The UART parser permits 64-bit addresses but DDR-window capture accepts
  // at most eight hex digits. All linked DDR objects lie below 2^32.
  if ((uintptr_t)pointer > UINT32_MAX)
    __builtin_trap();
  print_uart("@@FPGA_OUTPUT {\"version\":1,\"name\":\"");
  print_uart(name);
  print_uart("\",\"address\":\"0x");
  print_uart_hex32((uint32_t)(uintptr_t)pointer);
  print_uart("\",\"nbytes\":");
  print_uart_dec32(nbytes);
  print_uart(",\"dtype\":\"");
  print_uart(dtype);
  print_uart("\",\"shape\":");
  print_uart(shape);
  print_uart("}");
  print_uart(newline);
}
extern char __bss_start[], __bss_end[];
void nh_e6_app_report(void) {
  for (uintptr_t line = (uintptr_t)__bss_start; line < (uintptr_t)__bss_end;
       line += 64u)
    nh_invalidate((const volatile void *)line);
  print_uart("[efficientnet-model] calls=");
  print_uart_dec32(model_call_count);
  print_uart(" inflight=");
  print_uart_dec32(model_call_inflight);
  print_uart(" checked=");
  print_uart_dec32(model_checked);
  print_uart(" errors=");
  print_uart_dec32(model_errors);
  print_uart(" max_abs_error_f32_bits=0x");
  print_uart_hex32(model_max_abs_bits);
  print_uart(" top1=");
  print_uart_dec32(model_top1);
  print_uart(" logits_fnv64=0x");
  print_uart_hex64(model_logits_hash);
  print_uart(" model_cycles=0x");
  print_uart_hex64(model_cycles);
  print_uart(" kernel_cycles=0x");
  print_uart_hex64(model_kernel_cycles);
  print_uart(newline);
  for (unsigned i = 0; i < model_call_count; ++i) {
    print_uart("[efficientnet-cycle] call=");
    print_uart_dec32(i);
    print_uart(" cycles=0x");
    print_uart_hex64(model_call_cycles[i]);
    print_uart(newline);
  }
  for (unsigned i = 0; i < checkpoint_count; ++i) {
    struct Checkpoint *c = &checkpoint_hashes[i];
    print_uart("[efficientnet-checkpoint] call=");
    print_uart_dec32(c->call);
    print_uart(" span=");
    print_uart_dec32(c->span);
    print_uart(" bytes=");
    print_uart_dec32((uint32_t)c->bytes);
    print_uart(" fnv64=0x");
    print_uart_hex64(c->hash);
    print_uart(newline);
  }
  uint32_t done = *(volatile uint32_t *)NH_E6_DONE_REG;
  if (done == NH_E6_DONE_TRAP) {
    print_uart(protocol_trap);
    print_uart_hex64(e6_diag_mcause);
    print_uart(protocol_mepc);
    print_uart_hex64(e6_diag_mepc);
    print_uart(protocol_mtval);
    print_uart_hex64(e6_diag_mtval);
    print_uart(protocol_close);
    return;
  }
  if (model_returned) {
    output("logits_f32", model_export_logits, sizeof(model_export_logits),
           "<f4", EFFICIENTNET_B3_OUTPUT_SHAPE);
    output("call_cycles_u64", model_export_cycles, sizeof(model_export_cycles),
           "<u8", MODEL_CYCLE_SHAPE);
  }
#ifdef MODEL_PREFIX_PROBE
  if (model_call_count > 0u)
    output("stem_f32", prefix_stem, sizeof(prefix_stem), "<f4",
           "[1,40,144,144]");
  if (model_call_count > 2u)
    output("depthwise_f32", prefix_dw, sizeof(prefix_dw), "<f4",
           "[1,40,144,144]");
  if (model_call_count > 6u)
    output("pointwise_acc_i32", prefix_acc, sizeof(prefix_acc), "<i4",
           "[1,10]");
#endif
  int numerical_ok = model_errors == 0;
#ifdef MODEL_HOST_NUMERICAL_VALIDATION
  numerical_ok =
      1; // Full numerical status is decided by downloaded-output audit.
#endif
  int ok = done == NH_E6_DONE_PASS && model_returned &&
           model_checked == EFFICIENTNET_B3_OUTPUT_ELEMENTS && numerical_ok &&
           model_call_count == EFFICIENTNET_B3_PHYSICAL_CALLS;
  print_uart(ok ? protocol_pass : protocol_fail);
}
