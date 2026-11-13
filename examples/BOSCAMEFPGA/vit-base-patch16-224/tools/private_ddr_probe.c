// Physical private A/B/P visibility probe; generated kernel lives beside it.
#include "bosc_ame_dma.h"
#include "nh_e6_platform.h"
#include "uart.h"
#include <stddef.h>
#include <stdint.h>

extern void private_chain(void *, void *, int64_t, int64_t, int64_t, int64_t,
                          int64_t);
extern unsigned char __bosc_ame_private_ddr_start[];
static unsigned char output[256]
    __attribute__((section(".dma_source"), aligned(64), used));
static volatile uint32_t completed, errors, guard_errors, frame_errors;
static volatile uint32_t irq_errors;
static volatile uint32_t private_reason;
static volatile uint64_t private_address, private_bytes, read_sink;
static volatile uint64_t observed_hart, observed_mstatus, observed_sp;
static volatile uint64_t elapsed[3], hashes[3];

void *memcpy(void *destination, const void *source, size_t size) {
  unsigned char *dst = destination;
  const unsigned char *src = source;
  for (size_t i = 0; i < size; ++i)
    dst[i] = src[i];
  return destination;
}
void *memset(void *destination, int value, size_t size) {
  unsigned char *dst = destination;
  for (size_t i = 0; i < size; ++i)
    dst[i] = (unsigned char)value;
  return destination;
}
static uint64_t cycles(void) {
  uint64_t value;
  __asm__ volatile("rdcycle %0" : "=r"(value)::"memory");
  return value;
}
static uint64_t mie(void) {
  uint64_t value;
  __asm__ volatile("csrr %0, mstatus" : "=r"(value)::"memory");
  return value & 8u;
}
static void frame_failure(uint32_t reason, uint64_t address, uint64_t bytes) {
  private_reason = reason;
  private_address = address;
  private_bytes = bytes;
  __builtin_trap();
}

int e6_main(void) {
  bosc_ame_private_ddr_fail_hook = frame_failure;
  uint64_t value;
  __asm__ volatile("csrr %0, mhartid" : "=r"(value));
  observed_hart = value;
  __asm__ volatile("csrr %0, mstatus" : "=r"(value));
  observed_mstatus = value;
  uint64_t initial_mie = value & 8u;
  __asm__ volatile("mv %0, sp" : "=r"(value));
  observed_sp = value;
  nh_e6_progress(0x5100);
  nh_e6_cache_writes();
  // Isolate allocator entry and lifecycle before executing any AME.
  nh_e6_progress(0x5200);
  uint64_t preflight = bosc_ame_private_ddr_acquire(1536);
  irq_errors += mie() != initial_mie;
  nh_e6_progress(0x5201);
  bosc_ame_private_ddr_release(preflight, 1536);
  irq_errors += mie() != initial_mie;
  nh_e6_progress(0x5202);
  for (unsigned call = 0; call < 3; ++call) {
    const unsigned poison = (unsigned[]){0xa5, 0x5a, 0xe7}[call];
    nh_e6_cache_writes();
    memset(output, 0x39, sizeof(output));
    volatile unsigned char *frame = __bosc_ame_private_ddr_start + 64;
    for (unsigned i = 0; i < 1536; ++i)
      frame[i] = (unsigned char)poison;
    // Populate old cached payload/output lines before the device writes.
    uint64_t sink = 0;
    for (unsigned i = 0; i < 1536; i += 64)
      sink += frame[i];
    for (unsigned i = 0; i < sizeof(output); i += 64)
      sink += ((volatile unsigned char *)output)[i];
    read_sink = sink;
    nh_e6_progress(0x5300 + call);
    nh_e6_cache_writes();
    uint64_t begin = cycles();
    private_chain(output + 64, output + 64, 0, 2, 16, 16, 1);
    irq_errors += mie() != initial_mie;
    __asm__ volatile("fence iorw, iorw" ::: "memory");
    elapsed[call] = cycles() - begin;
    nh_e6_progress(0x5400 + call);
    nh_e6_cache_uncached_loads();
    const volatile int32_t *actual = (const volatile int32_t *)(output + 64);
    const volatile int32_t *p = (const volatile int32_t *)(frame + 1344);
    uint64_t hash = UINT64_C(14695981039346656037);
    for (unsigned i = 0; i < 32; ++i) {
      errors += actual[i] != -2671; // 17 + 2 * 64 * (-3) * 7.
      frame_errors += p[i] != -2671;
      for (unsigned byte = 0; byte < 4; ++byte) {
        hash ^= ((uint32_t)actual[i] >> (byte * 8)) & 255u;
        hash *= UINT64_C(1099511628211);
      }
    }
    for (unsigned i = 0; i < 128; ++i)
      frame_errors += frame[64 + i] != (unsigned char)-3;
    for (unsigned i = 0; i < 1024; ++i)
      frame_errors += frame[256 + i] != 7;
    for (unsigned i = 0; i < 64; ++i) {
      guard_errors += output[i] != 0x39 || output[192 + i] != 0x39;
      frame_errors += frame[i] != poison;
      frame_errors += frame[192 + i] != poison;
      frame_errors += frame[1280 + i] != poison;
      frame_errors += frame[1472 + i] != poison;
    }
    hashes[call] = hash;
    completed = call + 1;
    nh_e6_progress(0x5500 + call);
  }
  return errors || guard_errors || frame_errors || irq_errors ||
         bosc_ame_private_ddr_acquires != 4 ||
         bosc_ame_private_ddr_releases != 4;
}

const char *nh_e6_app_tag(void) { return "private-ddr-abp"; }
void probe_init_uart(uint32_t frequency, uint32_t baud) {
  init_uart(frequency, baud);
  print_uart("@@FPGA_START {\"version\":1}\r\n");
}
void nh_e6_app_report(void) {
  print_uart("[private-ddr] hart=0x");
  print_uart_hex64(observed_hart);
  print_uart(" mstatus=0x");
  print_uart_hex64(observed_mstatus);
  print_uart(" sp=0x");
  print_uart_hex64(observed_sp);
  print_uart(" irq_errors=");
  print_uart_dec32(irq_errors);
  print_uart("\r\n");
  for (unsigned call = 0; call < completed; ++call) {
    print_uart("[private-ddr] invocation=");
    print_uart_dec32(call);
    print_uart(" cycles=0x");
    print_uart_hex64(elapsed[call]);
    print_uart(" output_fnv64=0x");
    print_uart_hex64(hashes[call]);
    print_uart("\r\n");
  }
  print_uart("[private-ddr] completed=");
  print_uart_dec32(completed);
  print_uart(" errors=");
  print_uart_dec32(errors);
  print_uart(" guard_errors=");
  print_uart_dec32(guard_errors);
  print_uart(" frame_errors=");
  print_uart_dec32(frame_errors);
  print_uart(" acquires=0x");
  print_uart_hex64(bosc_ame_private_ddr_acquires);
  print_uart(" releases=0x");
  print_uart_hex64(bosc_ame_private_ddr_releases);
  print_uart(" allocator_failure=");
  print_uart_dec32(private_reason);
  print_uart(" address=0x");
  print_uart_hex64(private_address);
  print_uart(" bytes=0x");
  print_uart_hex64(private_bytes);
  print_uart("\r\n");
  uint32_t done = *(volatile uint32_t *)NH_E6_DONE_REG;
  if (done == NH_E6_DONE_TRAP) {
    print_uart(
        "@@FPGA_RESULT {\"version\":1,\"status\":\"trap\",\"mcause\":\"0x");
    print_uart_hex64(e6_diag_mcause);
    print_uart("\",\"mepc\":\"0x");
    print_uart_hex64(e6_diag_mepc);
    print_uart("\",\"mtval\":\"0x");
    print_uart_hex64(e6_diag_mtval);
    print_uart("\"}\r\n");
  } else {
    unsigned pass = done == NH_E6_DONE_PASS && completed == 3 && !errors &&
                    !irq_errors && !guard_errors && !frame_errors &&
                    bosc_ame_private_ddr_acquires == 4 &&
                    bosc_ame_private_ddr_releases == 4;
    print_uart(
        pass ? "@@FPGA_RESULT "
               "{\"version\":1,\"status\":\"exited\",\"app_exit_code\":0}\r\n"
             : "@@FPGA_RESULT "
               "{\"version\":1,\"status\":\"exited\",\"app_exit_code\":1}\r\n");
  }
}
