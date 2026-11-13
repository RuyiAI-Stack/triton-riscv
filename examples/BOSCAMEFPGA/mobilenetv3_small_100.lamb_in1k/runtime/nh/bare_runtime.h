//===- bare_runtime.h - Bare-metal C runtime interface --------------------===//
//
//===----------------------------------------------------------------------===//
//
// Lifecycle hooks for the NH bare-metal runtime on NR FPGA.
// _init() in bare_runtime.c calls them around main() in this order:
//   print_banner -> before_main -> main -> after_main.
// bare_runtime.c supplies weak no-op defaults; an application (e.g. hello.c)
// may provide strong definitions to customize banner / logging.
//
//===----------------------------------------------------------------------===//

#ifndef EXAMPLES_BOSCAMEFPGA_COMMON_RUNTIME_BARE_RUNTIME_H
#define EXAMPLES_BOSCAMEFPGA_COMMON_RUNTIME_BARE_RUNTIME_H

#include <stdint.h>

// Called from _init() in bare_runtime.c after UART init, before
// before_main()/main(). Override to print a boot banner.
void bare_runtime_print_banner(void);

// Called from _init() in bare_runtime.c immediately before main().
// Override to log entry into application code.
void bare_runtime_before_main(void);

// Called from _init() in bare_runtime.c after main() returns, before
// the final wfi park. Override to log completion.
void bare_runtime_after_main(void);

// Optional transport hooks, empty by default. Exit receives main's original
// return value after after_main; a fatal trap has no application return value.
void bare_runtime_report_start(void);
void bare_runtime_report_exit(int status);
void bare_runtime_report_trap(uintptr_t cause, uintptr_t pc, uintptr_t value);

#endif // EXAMPLES_BOSCAMEFPGA_COMMON_RUNTIME_BARE_RUNTIME_H
