#ifndef EFFNET_OPERATOR_SUPPORT_H
#define EFFNET_OPERATOR_SUPPORT_H

#include <stddef.h>
#include <stdint.h>

#ifdef HOST_TEST
#include <math.h>
#else
// Freestanding image: no libm. fabsf is in board_support.c; the rest are
// provided by runtime/boscame/nr/nr_math.c.
float fabsf(float);
float expf(float);
float sqrtf(float);
float erff(float);
float tanhf(float);
#endif

// Host oracles use 64 MiB. The NH/E6 image overrides this to 16 MiB so the
// arena stays inside link.ld's PMP window at 0x80100000..0x81100000.
#ifndef VIT_WORKSPACE_BYTES
#define VIT_WORKSPACE_BYTES (64u * 1024u * 1024u)
#endif
int launch(void);

#define MEMREF(R)                                                              \
  typedef struct {                                                             \
    void *allocated, *aligned;                                                 \
    int64_t offset;                                                            \
    int64_t sizes[R], strides[R];                                              \
  } MemRef##R
MEMREF(1);
MEMREF(2);
MEMREF(3);
MEMREF(4);

static inline MemRef1 make_1(void *p, int64_t a) {
  return (MemRef1){p, p, 0, {a}, {1}};
}
static inline MemRef2 make_2(void *p, int64_t a, int64_t b) {
  return (MemRef2){p, p, 0, {a, b}, {b, 1}};
}
static inline MemRef3 make_3(void *p, int64_t a, int64_t b, int64_t c) {
  return (MemRef3){p, p, 0, {a, b, c}, {b * c, c, 1}};
}
static inline MemRef4 make_4(void *p, int64_t a, int64_t b, int64_t c,
                             int64_t d) {
  return (MemRef4){p, p, 0, {a, b, c, d}, {b * c * d, c * d, d, 1}};
}

void *workspace(size_t byte_offset);
void nr_copy_bytes(void *destination, const void *source, size_t bytes);
int check_close(float actual, float expected, float atol, float rtol);
int print_check(const char *name, unsigned errors, float max_error);
void nr_puts(const char *s);
void nr_hex32(uint32_t x);
void nr_hex64(uint64_t x);
static inline size_t vit_align64(size_t byte_offset) {
  return (byte_offset + 63u) & ~(size_t)63;
}

// The host oracle reports clock() ticks through this legacy launch interface.
uint64_t nr_cycles(void);
// A non-host launcher must supply its own target synchronization
// implementation.
#ifndef HOST_TEST
void ame_fence(void);
#endif

#endif
