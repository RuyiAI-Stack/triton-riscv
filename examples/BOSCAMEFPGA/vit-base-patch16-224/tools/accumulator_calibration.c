//
// accumulator_calibration.c for BOSCAME FPGA model 'vit-base-patch16-224'.
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

// This header uses the UART declarations above.
#include "fpga_output.h"
#define K CALIBRATION_K
#define STRINGIFY_IMPL(x) #x
#define STRINGIFY(x) STRINGIFY_IMPL(x)
#define M 32
#define N 64
#ifndef CALIBRATION_STRESS
#define CALIBRATION_STRESS -1
#endif
#if CALIBRATION_STRESS < -1 || CALIBRATION_STRESS > 1
#error "CALIBRATION_STRESS must be -1, 0 or 1"
#endif
#if CALIBRATION_STRESS == -1
#define CALLS 16
#else
#define CALLS 8
#endif
#define FRAME_CALLS (CALLS / 2)
#define A_BYTES (M * K)
#define B_BYTES (N * K)
#define C_WORDS (M * K)
typedef struct {
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
} Desc;
extern void hot_ddr(int64_t, void *, int64_t, void *, int64_t, void *, int64_t,
                    void *);
extern void hot_spm(int64_t, void *, int64_t, void *, int64_t, void *, int64_t,
                    void *);
extern void hot_acc(int64_t, void *, int64_t, void *, int64_t, void *, int64_t,
                    void *);
extern const uint32_t bench_golden[];
extern unsigned char __bosc_ame_private_ddr_start[];
#define AREA __attribute__((section(".dma_source"), aligned(64), used))
static unsigned char a_store[A_BYTES + 128] AREA;
static unsigned char b_store[B_BYTES + 128] AREA;
static uint32_t c_store[C_WORDS + 32] AREA;
static uint32_t s_store[M * N + 32] AREA;
// Capture every completed invocation for independent daemon readmem checks.
// This ledger is populated after the measured adapter and is never rewritten.
//
uint32_t accumulator_calls[CALLS][2 * M * N] AREA;
typedef struct __attribute__((aligned(64))) {
  volatile uint64_t cycles, hash;
  volatile uint32_t errors, inputs, guards, frame_errors, completed;
  unsigned char pad[28];
} Slot;
static Slot results[CALLS];
static volatile uint32_t returned, validated, all_errors;
static volatile uint32_t captured_calls;
static volatile uint64_t warm_sink;
static uint64_t cycles(void) {
  uint64_t x;
  __asm__ volatile("rdcycle %0" : "=r"(x)::"memory");
  return x;
}
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
static int8_t av(unsigned r, unsigned q, unsigned stress) {
  static const int8_t x[5] = {127, -128, -1, 0, 1};
  return stress ? x[(r * 3 + q) % 5] : (int8_t)(r * 37 + q * 13 + 7);
}
static int8_t bv(unsigned n, unsigned q, unsigned stress) {
  static const int8_t x[5] = {-128, 127, 1, -1, 0};
  return stress ? x[(n * 7 + q * 3) % 5] : (int8_t)(n * 17 + q * 11 + 211);
}
static uint32_t oldc(unsigned index, unsigned stress) {
  return (stress ? 0x7fffff00u : 0x6a020000u) + index * 17u;
}
static void progress(unsigned call, unsigned phase) {
  nh_e6_progress(0x300000u + (call << 8) + phase);
}
static void initialize(unsigned call, unsigned stress) {
  nh_e6_cache_writes();
  progress(call, 1);
  memset(a_store, 0xa1, sizeof(a_store));
  memset(b_store, 0xb2, sizeof(b_store));
  memset(c_store, 0xc3, sizeof(c_store));
  memset(s_store, 0xd4, sizeof(s_store));
  progress(call, 2);
  int8_t *a = (int8_t *)(a_store + 64), *b = (int8_t *)(b_store + 64);
  for (unsigned r = 0; r < M; r++)
    for (unsigned q = 0; q < K; q++)
      a[r * K + q] = av(r, q, stress);
  for (unsigned n = 0; n < N; n++)
    for (unsigned q = 0; q < K; q++)
      b[n * K + q] = bv(n, q, stress);
  progress(call, 3);
  for (unsigned q = 0; q < C_WORDS; q++)
    c_store[16 + q] = oldc(q, stress);
  progress(call, 4);
  // Populate and cache every old private-P line before AME overwrites it.
  // These values intentionally differ from all zero-filled accumulator data.
  volatile uint64_t *frame =
      (volatile uint64_t *)(__bosc_ame_private_ddr_start + 64);
  for (unsigned q = 0; q < 8192 / 8; q++)
    frame[q] = 0x8d8d8d8d8d8d8d8dull ^ q;
  progress(call, 5);
  uint64_t sum = 0;
  for (unsigned q = 0; q < 8192 / 8; q += 8)
    sum ^= frame[q];
  progress(call, 6);
  // Warm external output cache lines as well as private storage.
  volatile uint32_t *c = c_store + 16;
  for (unsigned q = 0; q < C_WORDS; q += 16)
    sum ^= c[q];
  warm_sink = sum;
  progress(call, 7);
  __asm__ volatile("fence iorw,iorw" ::: "memory");
  progress(call, 8);
}
static unsigned inspect(unsigned call, unsigned stress, unsigned ddr) {
  nh_e6_cache_uncached_loads();
  Slot *s = &results[call];
  unsigned err = 0, in = 0, guards = 0, fe = 0;
  const int8_t *a = (const int8_t *)(a_store + 64),
               *b = (const int8_t *)(b_store + 64);
  const uint32_t *gold = bench_golden + stress * M * N;
  uint64_t hash = 14695981039346656037ull;
  for (unsigned r = 0; r < M; r++)
    for (unsigned n = 0; n < N; n++) {
      uint32_t actual = c_store[16 + r * K + n];
      err += actual != gold[r * N + n];
      err += s_store[16 + r * N + n] != gold[r * N + n];
      for (unsigned byte = 0; byte < 4; byte++) {
        hash ^= (actual >> (byte * 8)) & 255u;
        hash *= 1099511628211ull;
      }
    }
  for (unsigned r = 0; r < M; r++)
    for (unsigned q = N; q < K; q++)
      guards += c_store[16 + r * K + q] != oldc(r * K + q, stress);
  for (unsigned r = 0; r < M; r++)
    for (unsigned q = 0; q < K; q++)
      in += a[r * K + q] != av(r, q, stress);
  for (unsigned n = 0; n < N; n++)
    for (unsigned q = 0; q < K; q++)
      in += b[n * K + q] != bv(n, q, stress);
  for (unsigned q = 0; q < 64; q++) {
    guards += a_store[q] != 0xa1;
    guards += a_store[A_BYTES + 64 + q] != 0xa1;
    guards += b_store[q] != 0xb2;
    guards += b_store[B_BYTES + 64 + q] != 0xb2;
  }
  for (unsigned q = 0; q < 16; q++) {
    guards += c_store[q] != 0xc3c3c3c3u;
    guards += c_store[C_WORDS + 16 + q] != 0xc3c3c3c3u;
    guards += s_store[q] != 0xd4d4d4d4u;
    guards += s_store[M * N + 16 + q] != 0xd4d4d4d4u;
  }
  if (ddr) {
    volatile uint32_t *p =
        (volatile uint32_t *)(__bosc_ame_private_ddr_start + 64);
    for (unsigned q = 0; q < M * N; q++)
      fe += p[q] != gold[q];
  }
  s->hash = hash;
  s->errors = err;
  s->inputs = in;
  s->guards = guards;
  s->frame_errors = fe;
  __asm__ volatile("fence rw,rw" ::: "memory");
  s->completed = 1;
  return err + in + guards + fe;
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
  // The stack remains cached under both operand policies. Switch before
  // publishing the captured values into the DDR operand window.
  nh_e6_cache_writes();
  volatile uint32_t *ledger = accumulator_calls[call];
  for (unsigned q = 0; q < 2 * M * N; q++)
    ledger[q] = scratch[q];
  __asm__ volatile("fence iorw,iorw" ::: "memory");
  captured_calls = call + 1;
}
// This noinline typed adapter is the complete measured unit. All three
// variants produce C plus snapshot. The ACC-only raw function has one store;
// its second copy and cache-policy transition are explicitly timed here.
static __attribute__((noinline)) void invoke(unsigned route) {
  Desc a = {a_store + 64, a_store + 64, 0, {M, K}, {K, 1}};
  Desc b = {b_store + 64, b_store + 64, 0, {N, K}, {K, 1}};
  Desc c = {c_store + 16, c_store + 16, 0, {M, K}, {K, 1}};
  Desc snap = {s_store + 16, s_store + 16, 0, {M, N}, {N, 1}};
  if (route == 0)
    hot_ddr(2, &a, 2, &b, 2, &c, 2, &snap);
  else if (route == 1)
    hot_spm(2, &a, 2, &b, 2, &c, 2, &snap);
  else {
    uint32_t scratch[M * N];
    hot_acc(2, &a, 2, &b, 2, &c, 2, &snap);
    nh_e6_cache_uncached_loads();
    volatile const uint32_t *source = c_store + 16;
    for (unsigned r = 0; r < M; r++)
      for (unsigned n = 0; n < N; n++)
        scratch[r * N + n] = source[r * K + n];
    nh_e6_cache_writes();
    volatile uint32_t *dest = s_store + 16;
    for (unsigned q = 0; q < M * N; q++)
      dest[q] = scratch[q];
  }
}
static unsigned route_for(unsigned call) {
  static const unsigned routes[8] = {0, 1, 0, 1, 0, 2, 0, 2};
  return routes[call % 8];
}
static unsigned stress_for(unsigned call) {
#if CALIBRATION_STRESS == -1
  return call >= 8;
#else
  (void)call;
  return CALIBRATION_STRESS;
#endif
}
int e6_main(void) {
  bosc_ame_dma_fail_hook = nh_e6_publish_dma_failure;
  for (unsigned call = 0; call < CALLS; call++) {
    unsigned stress = stress_for(call), route = route_for(call),
             ddr = route == 0;
    nh_e6_progress(0x1100u + call);
    initialize(call, stress);
    progress(call, 0x10);
    uint64_t begin = cycles();
    invoke(route);
    __asm__ volatile("fence iorw,iorw" ::: "memory");
    results[call].cycles = cycles() - begin;
    progress(call, 0x11);
    returned = call + 1;
    all_errors += inspect(call, stress, ddr);
    progress(call, 0x12);
    capture(call);
    progress(call, 0x13);
    validated = call + 1;
    nh_e6_progress(0x2200u + call);
  }
  return all_errors != 0;
}
const char *nh_e6_app_tag(void) {
#if CALIBRATION_STRESS != -1
#if CALIBRATION_POST_ADD
  return "acc-path-post-add-k" STRINGIFY(CALIBRATION_TRIPS) "-stress" STRINGIFY(
      CALIBRATION_STRESS);
#else
  return "acc-path-k" STRINGIFY(CALIBRATION_TRIPS) "-stress" STRINGIFY(
      CALIBRATION_STRESS);
#endif
#else
#if CALIBRATION_POST_ADD
  return "acc-path-post-add-k" STRINGIFY(CALIBRATION_TRIPS);
#else
  return "acc-path-k" STRINGIFY(CALIBRATION_TRIPS);
#endif
#endif
}
void bench_init_uart(uint32_t freq, uint32_t baud) {
  init_uart(freq, baud);
  print_uart("@@FPGA_START {\"version\":1}\r\n");
}
void nh_e6_app_report(void) {
  for (unsigned q = 0; q < CALLS; q++)
    if (results[q].completed) {
      print_uart("[acc-path] invocation=");
      print_uart_dec32(q);
      unsigned route = route_for(q);
      print_uart(route == 0   ? " variant=ddr"
                 : route == 1 ? " variant=spm"
                              : " variant=acc");
      print_uart(" stress=");
      print_uart_dec32(stress_for(q));
      print_uart(" kernel_cycles=0x");
      print_uart_hex64(results[q].cycles);
      print_uart(" errors=");
      print_uart_dec32(results[q].errors);
      print_uart(" input_errors=");
      print_uart_dec32(results[q].inputs);
      print_uart(" guard_errors=");
      print_uart_dec32(results[q].guards);
      print_uart(" frame_errors=");
      print_uart_dec32(results[q].frame_errors);
      print_uart(" output_fnv64=0x");
      print_uart_hex64(results[q].hash);
      print_uart("\r\n");
    }
  print_uart("[acc-path] returned=");
  print_uart_dec32(returned);
  print_uart(" validated=");
  print_uart_dec32(validated);
  print_uart(" errors=");
  print_uart_dec32(all_errors);
  print_uart(" frame_acquires=0x");
  print_uart_hex64(bosc_ame_private_ddr_acquires);
  print_uart(" frame_releases=0x");
  print_uart_hex64(bosc_ame_private_ddr_releases);
  print_uart(" frame_peak_bytes=0x");
  print_uart_hex64(bosc_ame_private_ddr_peak_bytes);
  print_uart("\r\n");
  uint32_t done = *(volatile uint32_t *)NH_E6_DONE_REG;
  if (done == NH_E6_DONE_PASS && returned == CALLS && validated == CALLS &&
      captured_calls == CALLS && !all_errors &&
      bosc_ame_private_ddr_acquires == FRAME_CALLS &&
      bosc_ame_private_ddr_releases == FRAME_CALLS)
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
        done == NH_E6_DONE_PASS && returned == CALLS && validated == CALLS &&
                captured_calls == CALLS && !all_errors &&
                bosc_ame_private_ddr_acquires == FRAME_CALLS &&
                bosc_ame_private_ddr_releases == FRAME_CALLS
            ? "@@FPGA_RESULT "
              "{\"version\":1,\"status\":\"exited\",\"app_exit_code\":0}\r\n"
            : "@@FPGA_RESULT "
              "{\"version\":1,\"status\":\"exited\",\"app_exit_code\":1}\r\n");
}
