//
// spm_stale_c_probe.c for BOSCAME FPGA model 'vit-base-patch16-224'.
//
// Inputs: buffers, headers, or board state owned by this model tree.
//
// Outputs: the symbols the case harness or model runtime expects.
//
// When used: built with the model makefile under
// examples/BOSCAMEFPGA/vit-base-patch16-224.
//

#include "bosc_ame_dma.h"
#include "nh_e6_platform.h"
#include "uart.h"
#include <stddef.h>
#include <stdint.h>

#include "fpga_output.h"

// Correctness probe only: seed and cached control are outside target timing.
#define M 32
#define N 64
#define K 768
#define CALLS 4
#define C_WORDS (M * K)
#define AREA __attribute__((section(".dma_source"), aligned(64), used))
typedef struct {
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
} Desc;
extern void hot_spm(int64_t, void *, int64_t, void *, int64_t, void *, int64_t,
                    void *);
extern void ame_seed(int64_t, void *, int64_t, void *);
extern const uint32_t bench_golden[];
static unsigned char a_store[M * K + 128] AREA;
static unsigned char b_store[N * K + 128] AREA;
static uint32_t c_store[C_WORDS + 32] AREA;
static uint32_t s_store[M * N + 32] AREA;
static uint32_t seed_store[M * N + 32] AREA;
uint32_t accumulator_calls[CALLS][2 * M * N] AREA;
typedef struct __attribute__((aligned(64))) {
  volatile uint64_t cycles, hash;
  volatile uint32_t errors, inputs, guards, prewarm_errors;
  volatile uint32_t cached_stale, cached_current, cached_other, completed;
} Slot;
static Slot results[CALLS];
static volatile uint32_t returned, validated, captured, all_errors;
static volatile uint64_t warm_sink;

void *memcpy(void *d, const void *s, size_t n) {
  unsigned char *o = d;
  const unsigned char *i = s;
  for (size_t q = 0; q < n; q++)
    o[q] = i[q];
  return d;
}
void *memset(void *d, int c, size_t n) {
  unsigned char *o = d;
  for (size_t q = 0; q < n; q++)
    o[q] = (unsigned char)c;
  return d;
}
static uint64_t cycles(void) {
  uint64_t value;
  __asm__ volatile("rdcycle %0" : "=r"(value)::"memory");
  return value;
}
static int8_t av(unsigned r, unsigned q, unsigned stress) {
  static const int8_t x[5] = {127, -128, -1, 0, 1};
  return stress ? x[(r * 3 + q) % 5] : (int8_t)(r * 37 + q * 13 + 7);
}
static int8_t bv(unsigned n, unsigned q, unsigned stress) {
  static const int8_t x[5] = {-128, 127, 1, -1, 0};
  return stress ? x[(n * 7 + q * 3) % 5] : (int8_t)(n * 17 + q * 11 + 211);
}
static uint32_t oldc(unsigned q, unsigned stress) {
  return (stress ? 0x7fffff00u : 0x6a020000u) + q * 17u;
}
static uint32_t sentinel(unsigned call) {
  return 0x5555aaaau ^ (call * 0x11111111u);
}
static void progress(unsigned call, unsigned phase) {
  nh_e6_progress(0x400000u + (call << 8) + phase);
}
static void initialize(unsigned call, unsigned stress) {
  nh_e6_cache_writes();
  memset(a_store, 0xa1, sizeof(a_store));
  memset(b_store, 0xb2, sizeof(b_store));
  memset(c_store, 0xc3, sizeof(c_store));
  memset(s_store, 0xd4, sizeof(s_store));
  memset(seed_store, 0xe5, sizeof(seed_store));
  int8_t *a = (int8_t *)(a_store + 64), *b = (int8_t *)(b_store + 64);
  for (unsigned r = 0; r < M; r++)
    for (unsigned q = 0; q < K; q++)
      a[r * K + q] = av(r, q, stress);
  for (unsigned n = 0; n < N; n++)
    for (unsigned q = 0; q < K; q++)
      b[n * K + q] = bv(n, q, stress);
  for (unsigned q = 0; q < C_WORDS; q++)
    c_store[16 + q] = oldc(q, stress);
  for (unsigned r = 0; r < M; r++)
    for (unsigned n = 0; n < N; n++) {
      uint32_t correct = oldc(r * K + n, stress);
      seed_store[16 + r * N + n] = correct;
      c_store[16 + r * K + n] = correct ^ sentinel(call);
    }
  __asm__ volatile("fence iorw,iorw" ::: "memory");
}
static __attribute__((noinline)) void seed_and_control(unsigned call,
                                                       unsigned stress) {
  Slot *slot = &results[call];
  volatile const uint32_t *c = c_store + 16;
  uint64_t sum = 0;
  unsigned prewarm_errors = 0, stale = 0, current = 0, other = 0;
  // Keep write-policy cached loads through seed, control and target entry.
  for (unsigned r = 0; r < M; r++)
    for (unsigned n = 0; n < N; n++) {
      uint32_t value = c[r * K + n];
      prewarm_errors += value != (oldc(r * K + n, stress) ^ sentinel(call));
      sum ^= value;
    }
  warm_sink = sum;
  progress(call, 1);
  Desc seed = {seed_store + 16, seed_store + 16, 0, {M, N}, {N, 1}};
  Desc output = {c_store + 16, c_store + 16, 0, {M, K}, {K, 1}};
  // AME also dirties the exact bank0 P window with nonzero seed data.
  ame_seed(2, &seed, 2, &output);
  __asm__ volatile("fence iorw,iorw" ::: "memory");
  progress(call, 2);
  for (unsigned r = 0; r < M; r++)
    for (unsigned n = 0; n < N; n++) {
      uint32_t value = c[r * K + n], correct = oldc(r * K + n, stress);
      stale += value == (correct ^ sentinel(call));
      current += value == correct;
      other += value != correct && value != (correct ^ sentinel(call));
    }
  slot->prewarm_errors = prewarm_errors;
  slot->cached_stale = stale;
  slot->cached_current = current;
  slot->cached_other = other;
  progress(call, 3);
}
static __attribute__((noinline)) void invoke(void) {
  Desc a = {a_store + 64, a_store + 64, 0, {M, K}, {K, 1}};
  Desc b = {b_store + 64, b_store + 64, 0, {N, K}, {K, 1}};
  Desc c = {c_store + 16, c_store + 16, 0, {M, K}, {K, 1}};
  Desc snapshot = {s_store + 16, s_store + 16, 0, {M, N}, {N, 1}};
  hot_spm(2, &a, 2, &b, 2, &c, 2, &snapshot);
}
static unsigned inspect(unsigned call, unsigned stress) {
  nh_e6_cache_uncached_loads();
  Slot *slot = &results[call];
  const uint32_t *gold = bench_golden + stress * M * N;
  unsigned err = 0, inputs = 0, guards = 0;
  uint64_t hash = 14695981039346656037ull;
  for (unsigned r = 0; r < M; r++)
    for (unsigned n = 0; n < N; n++) {
      uint32_t value = c_store[16 + r * K + n];
      err += value != gold[r * N + n];
      err += s_store[16 + r * N + n] != gold[r * N + n];
      guards += seed_store[16 + r * N + n] != oldc(r * K + n, stress);
      for (unsigned byte = 0; byte < 4; byte++) {
        hash ^= (value >> (byte * 8)) & 255u;
        hash *= 1099511628211ull;
      }
    }
  for (unsigned r = 0; r < M; r++)
    for (unsigned q = N; q < K; q++)
      guards += c_store[16 + r * K + q] != oldc(r * K + q, stress);
  const int8_t *a = (const int8_t *)(a_store + 64),
               *b = (const int8_t *)(b_store + 64);
  for (unsigned r = 0; r < M; r++)
    for (unsigned q = 0; q < K; q++)
      inputs += a[r * K + q] != av(r, q, stress);
  for (unsigned n = 0; n < N; n++)
    for (unsigned q = 0; q < K; q++)
      inputs += b[n * K + q] != bv(n, q, stress);
  for (unsigned q = 0; q < 64; q++) {
    guards += a_store[q] != 0xa1 || a_store[M * K + 64 + q] != 0xa1;
    guards += b_store[q] != 0xb2 || b_store[N * K + 64 + q] != 0xb2;
  }
  for (unsigned q = 0; q < 16; q++) {
    guards +=
        c_store[q] != 0xc3c3c3c3u || c_store[C_WORDS + 16 + q] != 0xc3c3c3c3u;
    guards +=
        s_store[q] != 0xd4d4d4d4u || s_store[M * N + 16 + q] != 0xd4d4d4d4u;
    guards += seed_store[q] != 0xe5e5e5e5u ||
              seed_store[M * N + 16 + q] != 0xe5e5e5e5u;
  }
  slot->errors = err;
  slot->inputs = inputs;
  slot->guards = guards;
  slot->hash = hash;
  slot->completed = 1;
  return err + inputs + guards + slot->prewarm_errors + slot->cached_other;
}
static __attribute__((noinline)) void capture(unsigned call) {
  uint32_t scratch[2 * M * N];
  nh_e6_cache_uncached_loads();
  volatile const uint32_t *c = c_store + 16, *snapshot = s_store + 16;
  for (unsigned r = 0; r < M; r++)
    for (unsigned n = 0; n < N; n++) {
      scratch[r * N + n] = c[r * K + n];
      scratch[M * N + r * N + n] = snapshot[r * N + n];
    }
  nh_e6_cache_writes();
  volatile uint32_t *dest = accumulator_calls[call];
  for (unsigned q = 0; q < 2 * M * N; q++)
    dest[q] = scratch[q];
  __asm__ volatile("fence iorw,iorw" ::: "memory");
  captured = call + 1;
}
int e6_main(void) {
  bosc_ame_dma_fail_hook = nh_e6_publish_dma_failure;
  for (unsigned call = 0; call < CALLS; call++) {
    unsigned stress = call >= CALLS / 2;
    initialize(call, stress);
    seed_and_control(call, stress);
    progress(call, 0x10);
    uint64_t begin = cycles();
    invoke();
    __asm__ volatile("fence iorw,iorw" ::: "memory");
    results[call].cycles = cycles() - begin;
    returned = call + 1;
    progress(call, 0x11);
    all_errors += inspect(call, stress);
    capture(call);
    validated = call + 1;
    progress(call, 0x12);
  }
  return all_errors != 0;
}
const char *nh_e6_app_tag(void) { return "normal-spm-stale-c-k6"; }
void bench_init_uart(uint32_t freq, uint32_t baud) {
  init_uart(freq, baud);
  print_uart("@@FPGA_START {\"version\":1}\r\n");
}
void nh_e6_app_report(void) {
  unsigned exposed = 0;
  for (unsigned call = 0; call < CALLS; call++) {
    Slot *slot = &results[call];
    if (!slot->completed)
      continue;
    exposed += slot->cached_stale != 0;
    print_uart("[spm-stale-c] invocation=");
    print_uart_dec32(call);
    print_uart(" cached_stale=");
    print_uart_dec32(slot->cached_stale);
    print_uart(" cached_current=");
    print_uart_dec32(slot->cached_current);
    print_uart(" cached_other=");
    print_uart_dec32(slot->cached_other);
    print_uart(" prewarm_errors=");
    print_uart_dec32(slot->prewarm_errors);
    print_uart(" errors=");
    print_uart_dec32(slot->errors);
    print_uart(" input_errors=");
    print_uart_dec32(slot->inputs);
    print_uart(" guard_errors=");
    print_uart_dec32(slot->guards);
    print_uart(" output_fnv64=0x");
    print_uart_hex64(slot->hash);
    print_uart(" kernel_cycles=0x");
    print_uart_hex64(slot->cycles);
    print_uart("\r\n");
  }
  print_uart("[spm-stale-c] exposed_calls=");
  print_uart_dec32(exposed);
  print_uart(" returned=");
  print_uart_dec32(returned);
  print_uart(" validated=");
  print_uart_dec32(validated);
  print_uart(" errors=");
  print_uart_dec32(all_errors);
  print_uart("\r\n");
  uint32_t done = *(volatile uint32_t *)NH_E6_DONE_REG;
  unsigned pass = done == NH_E6_DONE_PASS && returned == CALLS &&
                  validated == CALLS && captured == CALLS && !all_errors &&
                  !bosc_ame_private_ddr_acquires &&
                  !bosc_ame_private_ddr_releases;
  if (pass)
    fpga_export_i32_2d("accumulator_calls", accumulator_calls, 0, CALLS,
                       2 * M * N, 2 * M * N, 1);
  if (done == NH_E6_DONE_TRAP) {
    print_uart(
        "@@FPGA_RESULT {\"version\":1,\"status\":\"trap\",\"mcause\":\"0x");
    print_uart_hex64(e6_diag_mcause);
    print_uart("\",\"mepc\":\"0x");
    print_uart_hex64(e6_diag_mepc);
    print_uart("\",\"mtval\":\"0x");
    print_uart_hex64(e6_diag_mtval);
    print_uart("\"}\r\n");
  } else
    print_uart(
        pass ? "@@FPGA_RESULT "
               "{\"version\":1,\"status\":\"exited\",\"app_exit_code\":0}\r\n"
             : "@@FPGA_RESULT "
               "{\"version\":1,\"status\":\"exited\",\"app_exit_code\":1}\r\n");
}
