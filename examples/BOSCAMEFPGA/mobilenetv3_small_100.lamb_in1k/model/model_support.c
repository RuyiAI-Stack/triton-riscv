//
// model_support.c for BOSCAME FPGA model 'mobilenetv3_small_100.lamb_in1k'.
//
// Inputs: buffers, headers, or board state owned by this model tree.
//
// Outputs: the symbols the case harness or model runtime expects.
//
// When used: built with the model makefile under
// examples/BOSCAMEFPGA/mobilenetv3_small_100.lamb_in1k.
//

#include "model_support.h"
#include "support.h"

#ifdef HOST_TEST
#include <stdio.h>
#include <stdlib.h>
#endif

// All execution and reporting happens on NH. Volatile keeps the in-flight
// context visible to a fatal trap on this same core; no cross-core mailbox or
// cache-policy change is needed for the scalar NH image.
static volatile struct {
  unsigned total, started, completed, current;
  unsigned returned, validation_executed, checked, errors;
  const char *name, *case_name, *symbol;
  uint64_t operator_ticks;
} state;

static void decimal(unsigned value) {
  char buffer[11];
  unsigned length = 0;
  do {
    buffer[length++] = (char)('0' + value % 10u);
    value /= 10u;
  } while (value);
  char text[11];
  for (unsigned i = 0; i < length; ++i)
    text[i] = buffer[length - 1u - i];
  text[length] = 0;
  nr_puts(text);
}

static void timer_field(void) {
#ifdef HOST_TEST
  nr_puts("ticks=0x");
#else
  nr_puts("cycles=0x");
#endif
}

static void op_identity(void) {
  nr_puts("[mobilenet-op] ");
  decimal(state.current);
  nr_puts("/");
  decimal(state.total);
  nr_puts(" symbol=");
  nr_puts(state.symbol ? state.symbol : "none");
  nr_puts(" case=");
  nr_puts(state.case_name ? state.case_name : "none");
  nr_puts(" name=");
  nr_puts(state.name ? state.name : "none");
}

void model_start(unsigned total) {
  state.total = total;
  state.started = state.completed = state.current = 0;
  state.returned = state.validation_executed = state.checked = state.errors = 0;
  state.name = state.case_name = state.symbol = 0;
  state.operator_ticks = 0;
  nr_puts("[mobilenet-model] BEGIN operators=");
  decimal(total);
#ifdef HOST_TEST
  nr_puts(" execution=host timer=clock-ticks\r\n");
#else
  nr_puts(" execution=nh-scalar-f32 timer=rdcycle\r\n");
#endif
}

void model_op_begin(unsigned index, const char *name, const char *case_name,
                    const char *symbol) {
  state.current = index;
  state.name = name;
  state.case_name = case_name;
  state.symbol = symbol;
  ++state.started;
  op_identity();
  nr_puts(" BEGIN\r\n");
}

void model_op_end(uint64_t elapsed) {
  ++state.completed;
  state.operator_ticks += elapsed;
  op_identity();
  nr_puts(" END ");
  timer_field();
  nr_hex64(elapsed);
  nr_puts("\r\n");
  state.current = 0;
}

void model_report_failure(void) {
  nr_puts("[mobilenet-model] FAIL in_flight=");
  decimal(state.current);
  nr_puts(" operators=");
  decimal(state.total);
  nr_puts(" started=");
  decimal(state.started);
  nr_puts(" completed=");
  decimal(state.completed);
  nr_puts(" kernel_returned=");
  decimal(state.returned);
  nr_puts(" validation_executed=");
  decimal(state.validation_executed);
  nr_puts(" checked=");
  decimal(state.checked);
  nr_puts(" errors=");
  decimal(state.errors);
  nr_puts("\r\n");
  if (state.current) {
    op_identity();
    nr_puts(" FAILED\r\n");
  }
}

static uint64_t logits_fnv64(const float *values, unsigned count) {
  const unsigned char *bytes = (const unsigned char *)values;
  uint64_t hash = UINT64_C(0xcbf29ce484222325);
  for (size_t i = 0; i < (size_t)count * sizeof(float); ++i) {
    hash ^= bytes[i];
    hash *= UINT64_C(0x100000001b3);
  }
  return hash;
}

int model_verify(const float *actual, const float *golden, unsigned count,
                 unsigned expected_top1, uint64_t elapsed) {
  unsigned errors = 0;
  float maximum = 0.0f;
  unsigned top[5] = {0, 0, 0, 0, 0};
  state.returned = 1;
  state.validation_executed = 1;
  if (state.started != state.total || state.completed != state.total ||
      count < 5)
    ++errors;
  for (unsigned i = 0; i < count; ++i) {
    int mismatch;
    if (!__builtin_isfinite(actual[i]) || !__builtin_isfinite(golden[i])) {
      maximum = __builtin_inff();
      mismatch = 1;
    } else {
      float difference = actual[i] - golden[i];
      if (difference < 0.0f)
        difference = -difference;
      if (difference > maximum)
        maximum = difference;
      mismatch = !check_close(actual[i], golden[i], 0.002f, 0.001f);
    }
    if (mismatch) {
      if (errors < 8) {
        union {
          float f;
          uint32_t u;
        } a = {actual[i]}, g = {golden[i]};
        nr_puts("[mobilenet-model] mismatch index=");
        decimal(i);
        nr_puts(" actual_f32_bits=0x");
        nr_hex32(a.u);
        nr_puts(" expected_f32_bits=0x");
        nr_hex32(g.u);
        nr_puts("\r\n");
      }
      ++errors;
    }
    state.checked = i + 1;
    state.errors = errors;
  }
  for (unsigned rank = 0; rank < 5; ++rank) {
    unsigned best = count;
    for (unsigned i = 0; i < count; ++i) {
      int used = 0;
      for (unsigned j = 0; j < rank; ++j)
        used |= top[j] == i;
      if (!used && (best == count || actual[i] > actual[best]))
        best = i;
    }
    top[rank] = best;
  }
  nr_puts("[mobilenet-model] kernel_returned=1 validation_executed=1 checked=");
  decimal(count);
  nr_puts(" errors=");
  // Add top-one agreement to the verdict before publishing errors.
  if (top[0] != expected_top1)
    ++errors;
  state.errors = errors;
  decimal(errors);
  nr_puts(" started=");
  decimal(state.started);
  nr_puts(" completed=");
  decimal(state.completed);
  nr_puts(" top1=");
  decimal(top[0]);
  nr_puts(" expected_top1=");
  decimal(expected_top1);
  union {
    float f;
    uint32_t u;
  } maximum_bits = {maximum};
  nr_puts(" max_abs_error_f32_bits=0x");
  nr_hex32(maximum_bits.u);
  nr_puts(" logits_fnv64=0x");
  nr_hex64(logits_fnv64(actual, count));
  nr_puts(" ");
  timer_field();
  nr_hex64(elapsed);
  nr_puts(" operator_");
  timer_field();
  nr_hex64(state.operator_ticks);
  nr_puts("\r\n[mobilenet-model] top5=");
  for (unsigned rank = 0; rank < 5; ++rank) {
    decimal(top[rank]);
    nr_puts(rank == 4 ? "\r\n" : ",");
  }
  if (top[0] != expected_top1) {
    nr_puts("model top1 mismatch expected=");
    nr_hex32(expected_top1);
    nr_puts("\r\n");
  }
#ifdef HOST_TEST
  const char *path = getenv("MODEL_LOGITS_FILE");
  if (path) {
    FILE *file = fopen(path, "wb");
    if (!file || fwrite(actual, sizeof(float), count, file) != count) {
      fputs("cannot write model logits\n", stderr);
      if (file)
        fclose(file);
      return 2;
    }
    if (fclose(file))
      return 2;
  }
#endif
  return print_check("mobilenetv3_small_100.lamb_in1k", errors, maximum);
}

#ifdef HOST_TEST
int main(void) { return launch(); }
#endif
