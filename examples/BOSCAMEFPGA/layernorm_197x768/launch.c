/* NH/E6 execution and independent padded-TTIR oracle for LayerNorm. */
#include <stddef.h>
#include <stdint.h>

#include "bosc_ame_dma.h"
#include "nh_e6_platform.h"
#include "uart.h"

#define ROWS 197u
#define WIDTH 768u
#define BLOCK 1024u
#define GUARD_WORDS 256u
#define GUARD UINT32_C(0x4f400001)
#define DIAG __attribute__((section(".bss.e6_diag")))

typedef struct {
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
} MemRef2;
typedef struct {
  void *allocated, *aligned;
  int64_t offset, size, stride;
} MemRef1;
typedef struct {
  uint32_t before[GUARD_WORDS];
  float values[ROWS * WIDTH];
  uint32_t after[GUARD_WORDS];
} Matrix;
typedef struct {
  uint32_t before[GUARD_WORDS];
  float values[WIDTH];
  uint32_t after[GUARD_WORDS];
} Affine;

Matrix layernorm_x __attribute__((section(".dma_source"), aligned(4096), used));
Affine layernorm_weight
    __attribute__((section(".dma_source"), aligned(4096), used));
Affine layernorm_bias
    __attribute__((section(".dma_source"), aligned(4096), used));
Matrix layernorm_out
    __attribute__((section(".dma_output"), aligned(4096), used));

volatile uint32_t layernorm_round DIAG;
volatile uint32_t layernorm_checked DIAG;
volatile uint32_t layernorm_errors DIAG;
volatile uint32_t layernorm_guard_errors DIAG;
volatile uint32_t layernorm_exact DIAG;
volatile uint32_t layernorm_max_abs_bits DIAG;
volatile uint64_t layernorm_cycles DIAG;

extern void _mlir_ciface_kernel_layernorm_197x768(MemRef2 *, MemRef1 *,
                                                  MemRef1 *, MemRef2 *);

static uint32_t bits(float value) {
  union {
    float f;
    uint32_t u;
  } v = {value};
  return v.u;
}
static float input(unsigned row, unsigned column, unsigned round) {
  int value = (int)((row * 37u + column * 13u + round * 71u) % 511u) - 255;
  return (float)value * 0.015625f + (float)(row % 7u) * 0.25f;
}
static float weight(unsigned column, unsigned round) {
  return (float)((int)((column * 11u + round * 19u) % 65u) - 24) * 0.0625f;
}
static float bias(unsigned column, unsigned round) {
  return (float)((int)((column * 7u + round * 31u) % 49u) - 24) * 0.03125f;
}
static void init_guards(uint32_t *before, uint32_t *after) {
  for (unsigned i = 0; i < GUARD_WORDS; ++i)
    before[i] = after[i] = GUARD;
}
static uint32_t check_guards(const uint32_t *before, const uint32_t *after) {
  uint32_t errors = 0;
  for (unsigned i = 0; i < GUARD_WORDS; ++i)
    errors += (before[i] != GUARD) + (after[i] != GUARD);
  return errors;
}
static void initialize(unsigned round) {
  init_guards(layernorm_x.before, layernorm_x.after);
  init_guards(layernorm_out.before, layernorm_out.after);
  init_guards(layernorm_weight.before, layernorm_weight.after);
  init_guards(layernorm_bias.before, layernorm_bias.after);
  for (unsigned column = 0; column < WIDTH; ++column) {
    layernorm_weight.values[column] = weight(column, round);
    layernorm_bias.values[column] = bias(column, round);
  }
  for (unsigned row = 0; row < ROWS; ++row)
    for (unsigned column = 0; column < WIDTH; ++column) {
      layernorm_x.values[row * WIDTH + column] = input(row, column, round);
      layernorm_out.values[row * WIDTH + column] = -12345.0f;
    }
}

/* Only complete one-dimensional vectors access SPM. Poison once before the
 * first launch; all 394 programs then share persistent, initially dirty banks.
 */
static void poison_spm(void) {
  for (unsigned bank = 0; bank < 2; ++bank) {
    uintptr_t base = bank ? UINT32_C(0x20400000) : UINT32_C(0x20000000);
    for (unsigned offset = 0; offset < 128u * 1024u; offset += 64)
      __asm__ volatile(".option push\n.option arch, +v\n"
                       "vsetivli zero, 16, e32, m1, ta, ma\n"
                       "vmv.v.x v0, %1\n"
                       "fence rw, rw\nvse32.v v0, (%0)\nfence rw, rw\n"
                       ".option pop\n"
                       :
                       : "r"(base + offset), "r"(GUARD)
                       : "memory");
  }
}

static uint32_t validate(unsigned round) {
  uint32_t errors = 0, exact = 0;
  float max_abs = 0.0f;
  for (unsigned row = 0; row < ROWS; ++row) {
    float sum = 0.0f;
    for (unsigned column = 0; column < WIDTH; ++column)
      sum += input(row, column, round);
    float mean = sum / (float)WIDTH;
    float squares = 0.0f;
    for (unsigned column = 0; column < WIDTH; ++column) {
      float x = input(row, column, round);
      float centered = x - mean;
      squares += centered * centered;
    }
    float inverse = 1.0f / __builtin_sqrtf(squares / (float)WIDTH + 1.0e-12f);
    for (unsigned column = 0; column < WIDTH; ++column) {
      float wanted = ((input(row, column, round) - mean) * inverse) *
                         weight(column, round) +
                     bias(column, round);
      float actual = layernorm_out.values[row * WIDTH + column];
      float delta = __builtin_fabsf(actual - wanted);
      if (delta > max_abs)
        max_abs = delta;
      errors += !(delta <= 2.0e-5f * (1.0f + __builtin_fabsf(wanted)));
      errors += bits(layernorm_x.values[row * WIDTH + column]) !=
                bits(input(row, column, round));
      exact += bits(actual) == bits(wanted);
    }
    layernorm_checked = (round * ROWS + row + 1) * WIDTH;
    nh_e6_progress(0x2000u + row);
  }
  for (unsigned column = 0; column < WIDTH; ++column) {
    errors +=
        bits(layernorm_weight.values[column]) != bits(weight(column, round));
    errors += bits(layernorm_bias.values[column]) != bits(bias(column, round));
  }
  layernorm_guard_errors +=
      check_guards(layernorm_x.before, layernorm_x.after) +
      check_guards(layernorm_out.before, layernorm_out.after) +
      check_guards(layernorm_weight.before, layernorm_weight.after) +
      check_guards(layernorm_bias.before, layernorm_bias.after);
  layernorm_exact += exact;
  if (max_abs > 0.0f && bits(max_abs) > layernorm_max_abs_bits)
    layernorm_max_abs_bits = bits(max_abs);
  return errors;
}

int e6_main(void) {
  bosc_ame_dma_fail_hook = nh_e6_publish_dma_failure;
  poison_spm();
  MemRef2 x = {
      layernorm_x.values, layernorm_x.values, 0, {ROWS, WIDTH}, {WIDTH, 1}};
  MemRef2 out = {
      layernorm_out.values, layernorm_out.values, 0, {ROWS, WIDTH}, {WIDTH, 1}};
  MemRef1 w = {layernorm_weight.values, layernorm_weight.values, 0, WIDTH, 1};
  MemRef1 b = {layernorm_bias.values, layernorm_bias.values, 0, WIDTH, 1};
  for (unsigned round = 0; round < 2; ++round) {
    layernorm_round = round;
    nh_e6_cache_writes();
    initialize(round);
    uint64_t begin, end;
    __asm__ volatile("rdcycle %0" : "=r"(begin));
    _mlir_ciface_kernel_layernorm_197x768(&x, &w, &b, &out);
    __asm__ volatile("rdcycle %0" : "=r"(end));
    layernorm_cycles += end - begin;
    nh_e6_cache_uncached_loads();
    uint32_t errors = validate(round);
    nh_e6_cache_writes();
    layernorm_errors += errors;
    if (layernorm_errors || layernorm_guard_errors)
      return 1;
  }
  return 0;
}

const char *nh_e6_app_tag(void) { return "layernorm-197x768-dma"; }
void nh_e6_app_report(void) {
  nh_e6_print_mhz_prefix();
  print_uart("checked=");
  print_uart_dec32(layernorm_checked);
  print_uart(" errors=");
  print_uart_dec32(layernorm_errors);
  print_uart(" guard_errors=");
  print_uart_dec32(layernorm_guard_errors);
  print_uart(" exact=");
  print_uart_dec32(layernorm_exact);
  print_uart(" max_abs_bits=0x");
  print_uart_hex32(layernorm_max_abs_bits);
  print_uart(" cycles=0x");
  print_uart_hex64(layernorm_cycles);
  print_uart(" round=");
  print_uart_dec32(layernorm_round);
  print_uart("\r\n");
}
