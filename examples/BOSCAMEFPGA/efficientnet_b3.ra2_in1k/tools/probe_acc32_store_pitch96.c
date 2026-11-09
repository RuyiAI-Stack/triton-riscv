/*
 * probe_acc32_store_pitch96.c
 *
 * Standalone micro-repro of the hanging Out store in
 *   conv2d_144x144_40x24_k1_s1
 *
 * What the production lowering actually emits for that store
 * (see BOSCAMETileLowering.cpp MatrixCopyOpConversion):
 *
 *   fence
 *   msettilem  M = rows            (= 16)
 *   msettilen  N = row_bytes / 4   (= 64/4 = 16)   // ACC32 "columns"
 *   msettype   mtype = 65602
 *   mlce32.m   acc0, (spm),  source_pitch   // SPM  -> ACC
 *   msce32.m   acc0, (ddr),  target_pitch   // ACC  -> DDR
 *   fence
 *
 * So the writeback is:
 *   - register class: ACC (acc0), NOT TR
 *   - TR0/TR1 are used earlier for MMA (mqma.b.mm acc0, tr0, tr1), then
 *     tile.store_acc dumps the result into SPM; the *boundary* store that
 *     hangs is a separate ACC32 scratch copy SPM->ACC->DDR.
 *
 * Geometry that hangs on NR FPGA (then-arm full BN tile):
 *   rows         = 16
 *   row_bytes    = 64     // 16 x i32
 *   source_pitch = 64     // SPM 16x16 i32, tightly packed
 *   target_pitch = 96     // DDR Out row = 24 x i32
 *   payload      = 16 * 64 = 1024 bytes moved through ACC once
 *
 * ACC capacity in the compiler contract is 16384 bytes
 * (matrixtransfer::kRegisterBytes); 1024 B fits easily. The programmed
 * tile here is only 16 x 16 ACC32 words, not a full 64x64-byte window.
 *
 * Build (host geometry check, no AME):
 *   cc -O2 -DPROBE_HOST_SIM=1 -o probe_host tools/probe_acc32_store_pitch96.c
 *   ./probe_host
 *
 * Build for NR board: link with the same board runtime as other cases and
 * call probe_acc32_store_pitch96() from launch (or make a tiny case that
 * only runs this). The AME path uses the same inline asm as the compiler.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum {
  ROWS = 16,
  ROW_BYTES = 64,             /* 16 x int32 */
  SRC_PITCH = 64,             /* SPM: packed 16x16 i32 tile */
  DST_PITCH = 96,             /* DDR: logical N=24 -> 24*4 */
  MTYPE = 65602,              /* matrixtransfer::kMtype (ACC32 raw-word mode) */
  PAYLOAD = ROWS * ROW_BYTES, /* 1024 */
};

/* Same operand packing as MatrixCopyOpConversion:
 *   $0 src, $1 dst, $2 rows, $3 row_bytes/4, $4 src_pitch, $5 dst_pitch, $6
 * mtype
 */
static inline void ame_acc32_rect_copy(void *src, void *dst, int64_t rows,
                                       int64_t row_bytes, int64_t src_pitch,
                                       int64_t dst_pitch, int64_t mtype) {
#if defined(__riscv) && !defined(PROBE_HOST_SIM)
  int64_t cols = row_bytes / 4; /* ACC32 column count */
  __asm__ __volatile__(
      "fence rw, rw\n\t"
      "msettilem zero, %[rows]\n\t"
      "msettilen zero, %[cols]\n\t"
      "msettype  zero, %[mtype]\n\t"
      "mlce32.m  acc0, (%[src]), %[sp]\n\t"
      "msce32.m  acc0, (%[dst]), %[dp]\n\t"
      "fence rw, rw"
      :
      : [src] "r"(src), [dst] "r"(dst), [rows] "r"(rows), [cols] "r"(cols),
        [sp] "r"(src_pitch), [dp] "r"(dst_pitch), [mtype] "r"(mtype)
      : "memory", "acc0");
#else
  (void)mtype;
  /* Host stand-in: byte-exact pitched rectangle copy (proves geometry). */
  for (int64_t r = 0; r < rows; ++r) {
    memcpy((uint8_t *)dst + r * dst_pitch, (uint8_t *)src + r * src_pitch,
           (size_t)row_bytes);
  }
#endif
}

int probe_acc32_store_pitch96(void) {
  /* 64B-aligned buffers: SPM-like src, DDR-like dst with pitch 96. */
  static uint8_t src[SRC_PITCH * ROWS] __attribute__((aligned(64)));
  static uint8_t dst[DST_PITCH * ROWS] __attribute__((aligned(64)));

  memset(dst, 0xA5, sizeof(dst));
  for (int r = 0; r < ROWS; ++r)
    for (int b = 0; b < ROW_BYTES; ++b)
      src[r * SRC_PITCH + b] = (uint8_t)(16 * r + b);

  printf("probe: ACC32 matrix store (NOT TR)\n");
  printf("  mlce32.m acc0 <- SPM  (pitch %d)\n", SRC_PITCH);
  printf("  msce32.m acc0 -> DDR  (pitch %d)  *** hang geometry ***\n",
         DST_PITCH);
  printf("  msettilem=%d  msettilen=%d (=row_bytes/4)  mtype=%d\n", ROWS,
         ROW_BYTES / 4, MTYPE);
  printf("  payload through ACC once = %d bytes (16 rows x 64 B)\n", PAYLOAD);
  printf("  DDR footprint spanned   = %d bytes ((rows-1)*96 + 64)\n",
         (ROWS - 1) * DST_PITCH + ROW_BYTES);

  ame_acc32_rect_copy(src, dst, ROWS, ROW_BYTES, SRC_PITCH, DST_PITCH, MTYPE);

#if defined(PROBE_HOST_SIM) || !defined(__riscv)
  /* Verify pitched layout on host. */
  for (int r = 0; r < ROWS; ++r) {
    for (int b = 0; b < ROW_BYTES; ++b) {
      if (dst[r * DST_PITCH + b] != src[r * SRC_PITCH + b]) {
        printf("FAIL: row %d byte %d\n", r, b);
        return 1;
      }
    }
    for (int b = ROW_BYTES; b < DST_PITCH; ++b) {
      if (dst[r * DST_PITCH + b] != 0xA5) {
        printf("FAIL: padding clobbered at row %d byte %d\n", r, b);
        return 1;
      }
    }
  }
  printf("HOST_SIM PASS: 1024 B placed as 16x64 into pitch-96 rows\n");
#else
  /* On board: if this returns, the ACC32 store completed (no hang). */
  printf("[effnet] ACC32 pitch96 store returned\n");
#endif
  return 0;
}

#if defined(PROBE_HOST_SIM) || !defined(__riscv)
int main(void) { return probe_acc32_store_pitch96(); }
#endif
