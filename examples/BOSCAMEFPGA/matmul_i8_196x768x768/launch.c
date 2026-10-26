/*
 * matmul_i8_196x768x768 launch: application startup, E6 driver and NH report.
 *
 * The image runs on the shared NH/E6 platform (runtime/boscame/nh_e6):
 * NH boots at 0x80000000, owns the UART and the verdict; E6 starts at
 * 0x80010000, and `nh_e6_e6_main()` runs the platform's address-space setup
 * (seven TOR PMP regions, `satp = 0`, `sfence.vma`) and the CSR 0xbc0 cache
 * policy *before* `e6_main()`.  That ordering is what makes the first AME
 * access in the kernel legal: the AME bus master reaches the DDR operand
 * windows through the PMP region bounded by
 * `__nh_e6_ddr_window_start/end`, which link.ld exports and places
 * `.dma_source`/`.dma_output` inside.
 *
 * This file does three things:
 *
 *   e6_main()          fill the operands, call the kernel through the
 *                      generated ABI bridge, then run the scalar INT32 oracle
 *                      and publish the diagnostics.
 *   nh_e6_app_tag()    report tag for the platform prefixes.
 *   nh_e6_app_report() NH-side report, printed on every outcome (PASS, FAIL,
 *                      TRAP, TIMEOUT), so a hang still produces evidence.
 *
 * The kernel keeps the Triton boundary masks of the reference kernel: M=196 is
 * not a multiple of BM=32, so the last M program masks the rows past M and the
 * masked store leaves excluded C elements untouched.  The launcher keeps the
 * logical [196,768] operand shapes and validates only those 196 rows.
 *
 * Progress markers (`nh_e6_progress`) are published at every phase boundary
 * and periodically inside the oracle: NH prints each new marker while it waits,
 * so the last marker on the console says exactly how far the run got.
 */
#include <stddef.h>
#include <stdint.h>

#include "bosc_ame_dma.h"
#include "matmul_i8_config.h"
#include "nh_e6_platform.h"
#include "uart.h"

/* ------------------------------------------------------------------ */
/* Problem size (the same numbers the TTIR and build.py use)           */
/* ------------------------------------------------------------------ */

#define M MATMUL_I8_M
#define N MATMUL_I8_N
#define K MATMUL_I8_K
#define BM MATMUL_I8_BM
#define BN MATMUL_I8_BN
#define BK MATMUL_I8_BK
#define GRID_M MATMUL_I8_GRID_M
#define GRID_N MATMUL_I8_GRID_N
#if BM > 64u
#error "the AME register file holds at most 64 rows per tile"
#endif
#if N % BN != 0u || K % BK != 0u
#error "the logical shape must divide evenly into the generated tile grid"
#endif
#define A_BYTES ((size_t)M * K)
#define B_BYTES ((size_t)N * K)
#define C_ELEMENTS ((size_t)M * N)
#define C_BYTES (C_ELEMENTS * 4u)

/* How much of C the on-board oracle re-computes.  The default checks every
 * logical row; `-DMATMUL_I8_CHECK_ROWS=16` (the debug image) checks the first
 * 16 rows plus the final partial BM tile (rows 192..195), which is what the
 * board capture window can afford for the full 115M multiply-add oracle. */
#ifndef MATMUL_I8_CHECK_ROWS
#define MATMUL_I8_CHECK_ROWS M
#endif
#if MATMUL_I8_CHECK_ROWS > M
#error "MATMUL_I8_CHECK_ROWS must not exceed M"
#endif
#if defined(MATMUL_I8_BOARD_CHECK) && MATMUL_I8_CHECK_ROWS != M
#error "MATMUL_I8_BOARD_CHECK requires a full-output oracle"
#endif

/* ------------------------------------------------------------------ */
/* ABI bridge interface (build.py --adapter experimental-boscame)      */
/* ------------------------------------------------------------------ */

typedef struct {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} MemRef2;

extern void MATMUL_I8_KERNEL(MemRef2 *a, MemRef2 *b, MemRef2 *c);

static inline MemRef2 make_2(void *pointer, int64_t rows, int64_t columns) {
  MemRef2 descriptor;
  descriptor.allocated = pointer;
  descriptor.aligned = pointer;
  descriptor.offset = 0;
  descriptor.sizes[0] = rows;
  descriptor.sizes[1] = columns;
  descriptor.strides[0] = columns;
  descriptor.strides[1] = 1;
  return descriptor;
}

/* ------------------------------------------------------------------ */
/* DDR operand windows, pinned by link.ld inside the PMP DDR window    */
/* ------------------------------------------------------------------ */

#if defined(MATMUL_I8_BOARD_CHECK)
/* Separate guard words from the payload without weakening its 64B alignment.
 * The public symbols and fixed payload offsets also let an independent
 * post-execution DDR checker inspect a full dump without using the E6
 * diagnostics. */
#define GUARD_WORDS 256u
#define GUARD_VALUE UINT32_C(0x6e5a1234)
typedef struct {
  uint32_t before[GUARD_WORDS];
  int8_t elements[A_BYTES];
  uint32_t after[GUARD_WORDS];
} AStorage;
typedef struct {
  uint32_t before[GUARD_WORDS];
  int8_t elements[B_BYTES];
  uint32_t after[GUARD_WORDS];
} BStorage;
typedef struct {
  uint32_t before[GUARD_WORDS];
  uint32_t elements[C_ELEMENTS];
  uint32_t after[GUARD_WORDS];
} CStorage;
AStorage a_storage __attribute__((section(".dma_source"), aligned(4096), used));
BStorage b_storage __attribute__((section(".dma_source"), aligned(4096), used));
CStorage c_storage __attribute__((section(".dma_output"), aligned(4096), used));
#define a_matrix a_storage.elements
#define b_matrix b_storage.elements
#define c_matrix ((int32_t *)c_storage.elements)
#else
static int8_t a_matrix[A_BYTES]
    __attribute__((section(".dma_source"), aligned(4096), used));
static int8_t b_matrix[B_BYTES]
    __attribute__((section(".dma_source"), aligned(4096), used));
static int32_t c_matrix[C_ELEMENTS]
    __attribute__((section(".dma_output"), aligned(4096), used));
#endif

/* ------------------------------------------------------------------ */
/* Diagnostics read by nh_e6_app_report()                              */
/* ------------------------------------------------------------------ */

#define DIAG __attribute__((section(".bss.e6_diag")))

volatile uint32_t e6_diag_phase DIAG;
volatile uint32_t e6_diag_errors DIAG;
volatile uint32_t e6_diag_max_abs DIAG;
volatile uint32_t e6_diag_checked_rows DIAG;
volatile uint64_t e6_diag_cycles DIAG;
#if defined(MATMUL_I8_BOARD_CHECK)
volatile uint32_t e6_diag_round DIAG;
volatile uint32_t e6_diag_guard_errors DIAG;
#endif
#if defined(MATMUL_I8_TRACE)
volatile int32_t e6_diag_first_got[8] DIAG;
volatile int32_t e6_diag_first_want[8] DIAG;
#endif

/*
 * Phase boundary: publish the marker through the platform, then re-select the
 * write-through store policy the delivered E6 needs for its pending writes.
 * The PMP regions and `satp = 0` this policy depends on were installed by
 * `nh_e6_e6_main()` before `e6_main()`.
 */
static inline void e6_stage(uint32_t value) {
  e6_diag_phase = value;
  nh_e6_progress(value);
  nh_e6_cache_writes();
}

static inline uint64_t cycles(void) {
  uint64_t value;
  __asm__ volatile("rdcycle %0" : "=r"(value));
  return value;
}

/* The kernel accumulates into C (`mqma.b.mm` has += semantics), so C starts at
 * the same deterministic pattern the oracle expects. */
#if defined(MATMUL_I8_BOARD_CHECK)
/* All arithmetic in the generators is bounded. Product accumulation in the
 * oracle uses uint32_t to model the kernel's exact INT32 wrap semantics.
 * B is physically [N,K], and the independent post-execution checker models
 * that layout. */
static int32_t board_a(unsigned i, unsigned q, unsigned round) {
  return (int32_t)((i * 13u + q * 7u + round * 71u) % 256u) - 128;
}
static int32_t board_b(unsigned j, unsigned q, unsigned round) {
  return (int32_t)((j * 11u + q * 5u + round * 37u) % 256u) - 128;
}
static uint32_t board_c(unsigned i, unsigned j, unsigned round) {
  return UINT32_C(0x7fffff00) + (i * 3u + j + round * 97u) % 512u;
}

static void fill_operands(unsigned round) {
  for (unsigned g = 0; g < GUARD_WORDS; ++g) {
    a_storage.before[g] = a_storage.after[g] = GUARD_VALUE;
    b_storage.before[g] = b_storage.after[g] = GUARD_VALUE;
    c_storage.before[g] = c_storage.after[g] = GUARD_VALUE;
  }
  for (unsigned i = 0; i < M; ++i)
    for (unsigned q = 0; q < K; ++q)
      a_matrix[(size_t)i * K + q] = (int8_t)board_a(i, q, round);
  for (unsigned j = 0; j < N; ++j)
    for (unsigned q = 0; q < K; ++q)
      b_matrix[(size_t)j * K + q] = (int8_t)board_b(j, q, round);
  for (unsigned i = 0; i < M; ++i)
    for (unsigned j = 0; j < N; ++j)
      c_matrix[(size_t)i * N + j] = board_c(i, j, round);
}

static uint32_t validate_guards_and_inputs(unsigned round) {
  uint32_t errors = 0;
  for (unsigned g = 0; g < GUARD_WORDS; ++g) {
    errors += a_storage.before[g] != GUARD_VALUE;
    errors += a_storage.after[g] != GUARD_VALUE;
    errors += b_storage.before[g] != GUARD_VALUE;
    errors += b_storage.after[g] != GUARD_VALUE;
    errors += c_storage.before[g] != GUARD_VALUE;
    errors += c_storage.after[g] != GUARD_VALUE;
  }
  for (unsigned i = 0; i < M; ++i)
    for (unsigned q = 0; q < K; ++q)
      errors += a_matrix[(size_t)i * K + q] != board_a(i, q, round);
  for (unsigned j = 0; j < N; ++j)
    for (unsigned q = 0; q < K; ++q)
      errors += b_matrix[(size_t)j * K + q] != board_b(j, q, round);
  return errors;
}

static uint32_t validate(unsigned round) {
  uint32_t errors = 0;
  for (unsigned i = 0; i < M; ++i) {
    for (unsigned j = 0; j < N; ++j) {
      uint32_t want = board_c(i, j, round);
      for (unsigned q = 0; q < K; ++q)
        want += (uint32_t)(board_a(i, q, round) * board_b(j, q, round));
      uint32_t got = (uint32_t)c_matrix[(size_t)i * N + j];
#if defined(MATMUL_I8_TRACE)
      if (i == 0u && j < 8u) {
        e6_diag_first_got[j] = (int32_t)got;
        e6_diag_first_want[j] = (int32_t)want;
      }
#endif
      errors += got != want;
    }
    ++e6_diag_checked_rows;
    /* Two full grids share the same SPM contents; only DDR operands are
     * reinitialized between them. Distinct markers identify the oracle row. */
    nh_e6_progress(0x2000u + round * M + i);
  }
  return errors;
}
#else
static void fill_operands(void) {
  for (unsigned i = 0; i < M; ++i)
    for (unsigned q = 0; q < K; ++q)
      a_matrix[(size_t)i * K + q] = (int8_t)(((i * 13u + q * 7u) % 31u) - 15);
  for (unsigned j = 0; j < N; ++j)
    for (unsigned q = 0; q < K; ++q)
      b_matrix[(size_t)j * K + q] = (int8_t)(((j * 11u + q * 5u) % 29u) - 14);
  for (unsigned i = 0; i < M; ++i)
    for (unsigned j = 0; j < N; ++j)
      c_matrix[(size_t)i * N + j] = (int32_t)((i * 3u + j) % 7u) - 3;
}

/* C[i,j] = previous C[i,j] + sum_q A[i,q] * B[j,q] (B is the physical
 * [N,K] tile, exactly what MQMA.B.MM expects). */
static uint32_t validate(void) {
  uint32_t errors = 0;
  int32_t max_abs = 0;
  unsigned checked = 0;
  for (unsigned i = 0; i < M; ++i) {
    if (i >= MATMUL_I8_CHECK_ROWS && i < M - (M % BM))
      continue;
    checked++;
    for (unsigned j = 0; j < N; ++j) {
      int32_t want = (int32_t)((i * 3u + j) % 7u) - 3;
      for (unsigned q = 0; q < K; ++q)
        want += (int32_t)a_matrix[(size_t)i * K + q] *
                (int32_t)b_matrix[(size_t)j * K + q];
      int32_t got = c_matrix[(size_t)i * N + j];
#if defined(MATMUL_I8_TRACE)
      if (i == 0u && j < 8u) {
        e6_diag_first_got[j] = got;
        e6_diag_first_want[j] = want;
      }
#endif
      int32_t delta = got - want;
      if (delta < 0)
        delta = -delta;
      if (delta > max_abs)
        max_abs = delta;
      if (got != want)
        errors++;
    }
    e6_diag_checked_rows = checked;
    /* Keep the platform's progress report moving during the long check. */
    if ((i & 7u) == 0u)
      nh_e6_progress(0x200u + (i >> 3));
    if (errors != 0u)
      break;
  }
  e6_diag_max_abs = (uint32_t)max_abs;
  return errors;
}
#endif

/* ------------------------------------------------------------------ */
/* E6 entry                                                            */
/* ------------------------------------------------------------------ */

int e6_main(void) {
  /* A failed transfer records channel/reason/STATUS/token and traps; the E6
   * trap entry publishes 0xed and NH prints the failure context. */
  bosc_ame_dma_fail_hook = nh_e6_publish_dma_failure;

#if defined(MATMUL_I8_BOARD_CHECK)
  e6_diag_errors = 0;
  e6_diag_guard_errors = 0;
  e6_diag_checked_rows = 0;
  e6_diag_cycles = 0;
  for (unsigned round = 0; round < 2u; ++round) {
    e6_diag_round = round;
    e6_stage(0x11u + round);
    fill_operands(round);
    __asm__ volatile("fence rw, rw" ::: "memory");
    MemRef2 a = make_2(a_matrix, M, K);
    MemRef2 b = make_2(b_matrix, N, K);
    MemRef2 c = make_2(c_matrix, M, N);
    e6_stage(0x30u + round * 2u);
    uint64_t begin = cycles();
    MATMUL_I8_KERNEL(&a, &b, &c);
    e6_diag_cycles += cycles() - begin;
    __asm__ volatile("fence rw, rw" ::: "memory");
    e6_stage(0x31u + round * 2u);
    nh_e6_cache_uncached_loads();
    e6_diag_guard_errors += validate_guards_and_inputs(round);
    e6_diag_errors += validate(round);
    __asm__ volatile("fence rw, rw" ::: "memory");
  }
  uint32_t errors = e6_diag_errors + e6_diag_guard_errors;
  e6_stage(errors == 0u ? 0x02u : 0xeeu);
  return errors == 0u ? 0 : 1;
#else
  e6_stage(0x11u);
  /* The operands keep their logical shape; the kernel masks the boundary
   * tile's rows past M. */
  fill_operands();
  __asm__ volatile("fence rw, rw" ::: "memory");

  MemRef2 a = make_2(a_matrix, M, K);
  MemRef2 b = make_2(b_matrix, N, K);
  /* B is physically [N,K], so its descriptor has strides [K,1]. The
   * compiler's logical [K,N] AME view is built inside the kernel. */
  MemRef2 c = make_2(c_matrix, M, N);

  e6_stage(0x30u);
  uint64_t begin = cycles();
  MATMUL_I8_KERNEL(&a, &b, &c);
  e6_diag_cycles = cycles() - begin;
  __asm__ volatile("fence rw, rw" ::: "memory");
  e6_stage(0x31u);

  /* Read the AME-updated C with the reference's uncached-load policy. */
  nh_e6_cache_uncached_loads();
  uint32_t errors = validate();
  e6_diag_errors = errors;
  __asm__ volatile("fence rw, rw" ::: "memory");
  e6_stage(errors == 0u ? 0x02u : 0xeeu);

  return errors == 0u ? 0 : 1;
#endif
}

/* ------------------------------------------------------------------ */
/* NH report                                                           */
/* ------------------------------------------------------------------ */

/* The tag identifies the image variant on the console; the tester selects the
 * image by file name, not by tag. */
#if defined(MATMUL_I8_BOARD_CHECK)
const char *nh_e6_app_tag(void) { return "matmul-i8-196x768x768-boardcheck"; }
#else
const char *nh_e6_app_tag(void) { return "matmul-i8-196x768x768"; }
#endif

void nh_e6_app_report(void) {
  nh_e6_print_mhz_prefix();
  print_uart("stats phase=0x");
  print_uart_hex32(e6_diag_phase);
  print_uart(" cycles=0x");
  print_uart_hex64(e6_diag_cycles);
  print_uart(" checked_rows=");
  print_uart_dec32(e6_diag_checked_rows);
  print_uart(" errors=");
  print_uart_dec32(e6_diag_errors);
#if defined(MATMUL_I8_BOARD_CHECK)
  print_uart(" guard_errors=");
  print_uart_dec32(e6_diag_guard_errors);
  print_uart(" round=");
  print_uart_dec32(e6_diag_round);
#endif
  print_uart(" max_abs=");
  print_uart_dec32(e6_diag_max_abs);
  print_uart("\r\n");
#if defined(MATMUL_I8_TRACE)
  for (unsigned j = 0; j < 8u; ++j) {
    nh_e6_print_mhz_prefix();
    print_uart("sample column=");
    print_uart_dec32(j);
    print_uart(" got=0x");
    print_uart_hex32((uint32_t)e6_diag_first_got[j]);
    print_uart(" want=0x");
    print_uart_hex32((uint32_t)e6_diag_first_want[j]);
    print_uart("\r\n");
  }
#endif
}
