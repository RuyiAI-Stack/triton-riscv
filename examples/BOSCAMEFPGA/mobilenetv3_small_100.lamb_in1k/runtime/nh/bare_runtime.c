//===- bare_runtime.c - Bare-metal C runtime for NR FPGA ------------------===//
//
//===----------------------------------------------------------------------===//
//
// Bare-metal C runtime for the NH core on the NR FPGA platform.
// Provides _init (UART bring-up, lifecycle hooks around main, then wfi)
// and the default machine-mode trap handler used by the hello example.
//
//===----------------------------------------------------------------------===//

#include "bare_runtime.h"
#include "uart.h"

#include <stdint.h>

// Default scheduler-pool clock (5.5296 MHz) and UART baud (115200).
// uart.c computes the divisor from these values (3 for the default clock).
// Override with -DUART_FREQ_HZ / -DUART_BAUD to match another board profile.
#ifndef UART_FREQ_HZ
#define UART_FREQ_HZ 5529600u
#endif

#ifndef UART_BAUD
#define UART_BAUD 115200u
#endif

// Default weak no-op implementations of the bare-runtime lifecycle hooks.
// _init() calls them in order: print_banner -> before_main -> main ->
// after_main. An application may provide strong definitions of the same
// symbols (see hello.c) to print a banner or log around main(); if none
// are linked, these empty stubs keep the CRT path valid with no extra I/O.
__attribute__((weak)) void bare_runtime_print_banner(void) {}

__attribute__((weak)) void bare_runtime_before_main(void) {}

__attribute__((weak)) void bare_runtime_after_main(void) {}

// Optional transport hooks. A scheduler can link strong definitions to report
// the original main return value after all application diagnostics are printed.
__attribute__((weak)) void bare_runtime_report_start(void) {}
__attribute__((weak)) void bare_runtime_report_exit(int status) {
  (void)status;
}
__attribute__((weak)) void
bare_runtime_report_trap(uintptr_t cause, uintptr_t pc, uintptr_t value) {
  (void)cause;
  (void)pc;
  (void)value;
}

// Default machine-mode trap handler, called from crt_uart.S trap_entry.
//
// Calling convention (set up by the assembly stub):
//   mcause - exception/interrupt cause (CSR mcause -> a0)
//   mepc   - faulting PC (CSR mepc -> a1)
//   regs   - pointer to the stacked x1-x31 frame on the trap stack (a2);
//            unused here, but available for a recovering override
//
// The CRT expects the return value to be written back into mepc before
// mret. This default implementation never returns: it dumps mcause/mepc/
// mtval over UART and parks the hart in wfi. Marked weak so an app can
// supply a strong handle_trap that resumes (return a new mepc) instead.
__attribute__((weak)) uintptr_t handle_trap(uintptr_t mcause, uintptr_t mepc,
                                            uintptr_t regs) {
  // Record CSR and caller context before UART calls. The assembly frame is
  // 272 bytes below the interrupted stack pointer.
  uintptr_t mtval;
  __asm__ volatile("csrr %0, mtval" : "=r"(mtval));

  print_uart("\r\nTRAP mcause=0x");
  print_uart_addr((uint64_t)mcause);
  print_uart(" mepc=0x");
  print_uart_addr((uint64_t)mepc);
  print_uart(" mtval=0x");
  print_uart_addr((uint64_t)mtval);
  print_uart(" sp=0x");
  print_uart_addr((uint64_t)(regs + 272u));
  print_uart(" ra=0x");
  print_uart_addr(((const uintptr_t *)regs)[1]);
  print_uart("\r\n");

  // Application diagnostics precede the terminal protocol record. The model
  // and standalone operator images each link only their own optional hook.
  extern void model_report_failure(void) __attribute__((weak));
  extern void nr_report_failure(void) __attribute__((weak));
  if (model_report_failure)
    model_report_failure();
  if (nr_report_failure)
    nr_report_failure();
  bare_runtime_report_trap(mcause, mepc, mtval);

  // Fatal path: there is no OS/scheduler to resume into after a trap dump.
  // Spin in wfi so the hart stays idle and UART output is not flooded.
  while (1)
    __asm__ volatile("wfi");
}

// C entry point jumped to by crt_uart.S after BSS/TBSS clear and stack/gp
// setup. Initializes UART, runs the weak lifecycle hooks around main(), then
// parks the hart in wfi. Never returns: bare metal has no host process to
// exit back to, and returning would fall off into undefined code.
void _init(void) {
  init_uart(UART_FREQ_HZ, UART_BAUD);

  bare_runtime_report_start();

  bare_runtime_print_banner();

  extern int main(void);
  bare_runtime_before_main();
  int status = main();
  bare_runtime_after_main();
  bare_runtime_report_exit(status);

  while (1)
    __asm__ volatile("wfi");
}
