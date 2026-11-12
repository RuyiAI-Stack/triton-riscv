# NH-only runtime

The following seven files are copied from
[CBalaa/fpga-scheduler](https://github.com/CBalaa/fpga-scheduler/tree/a2d5c70ee6dc7b3c5b5719d62a0ac5d76b8aa927/runtime/bosc),
commit `a2d5c70ee6dc7b3c5b5719d62a0ac5d76b8aa927`:

- `crt_uart.S`
- `bare_runtime.c`
- `bare_runtime.h`
- `encoding.h`
- `uart.c`
- `uart.h`
- `uart_protocol.c`

Their original license notices are retained. These files are provided under
Apache-2.0; see [NOTICE](NOTICE) for provenance and the repository's
[LICENSE](../../../../../LICENSE). `audit.py` is a separate local validation
tool added for the MobileNet integration.

The local trap handler also prints the interrupted stack pointer and return
address, and calls optional application diagnostics before the terminal UART
record. These additions are local changes to the copied runtime.

`crt_uart.S` starts NH hart 0 at `0x80000000`, enables scalar floating point,
initializes the stack and BSS, then enters `bare_runtime.c`. The C runtime
initializes UART at 115200 baud with a 5.5296 MHz clock, emits the scheduler
UART v1 START record, calls `int main(void)`, reports its original return code,
and waits in `wfi`. Fatal traps emit a UART v1 trap record. E6 is not released;
the runtime does not start DMA or AME.

Link `crt_uart.o`, `bare_runtime.o`, `uart.o` and `uart_protocol.o` with the
application. The linker script must keep `.text.init` first and define
`__global_pointer$`, `__stack_top`, `__bss_start`, `__bss_end`, `__tbss_start`
and `__tbss_end`. Operator builds use `../../link_nh.ld`; the complete model
uses its own layout. Compile kernels and application code for scalar
`rv64imafdc_zicsr_zifencei`, with LLVM vector target features disabled.

Before converting an ELF to `.bin`, run:

```bash
python3 audit.py application.elf --objdump /path/to/llvm-objdump --output audit.json
```

The audit checks the RISC-V ELF entry, scheduler protocol markers, and absence
of E6 startup, vector instructions and undecoded instructions. It is a static
compatibility check; board execution and numerical validation remain separate.
