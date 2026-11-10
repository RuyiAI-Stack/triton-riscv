// Host-side support symbols for the generated whole-model adapters.
//
// The adapters' numerical kernel calls are replaced by the Python oracle
// dispatcher; only the NR/E6 diagnostic and timing helpers remain.  They are
// no-ops here so the generated C can be linked into an x86 shared library.
//
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "nr_runtime.h"
#include "support.h"

void nr_puts(const char *s) { (void)s; }
void nr_hex32(uint32_t x) { (void)x; }
void nr_hex64(uint64_t x) { (void)x; }
uint64_t nr_cycles(void) { return 0; }

void nr_diag_reset(uint64_t position) { (void)position; }
void nr_diag_mark(uint64_t stage, uint64_t detail) {
  (void)stage;
  (void)detail;
}
void nr_diag_memref(unsigned operand, uintptr_t descriptor, uintptr_t aligned,
                    int64_t offset, int64_t rows, int64_t cols, int64_t stride0,
                    int64_t stride1) {
  (void)operand;
  (void)descriptor;
  (void)aligned;
  (void)offset;
  (void)rows;
  (void)cols;
  (void)stride0;
  (void)stride1;
}

void *workspace(size_t byte_offset) {
  (void)byte_offset;
  return NULL;
}
void nr_copy_bytes(void *destination, const void *source, size_t bytes) {
  if (bytes)
    memcpy(destination, source, bytes);
}
int check_close(float actual, float expected, float atol, float rtol) {
  (void)actual;
  (void)expected;
  (void)atol;
  (void)rtol;
  return 0;
}
int print_check(const char *name, unsigned errors, float max_error) {
  (void)name;
  (void)errors;
  (void)max_error;
  return 0;
}
void ame_fence(void) {}
