OUTPUT_ARCH("riscv")
ENTRY(_start)

/* NH/E6 image layout for the INT8 matmul.  Boot, the dual-core release and
 * the platform verdict live in runtime/boscame/nh_e6; this script only fixes
 * where the image and the DDR operand windows go.
 *
 * The operand sections must be inside the DDR window the platform exports
 * (`__nh_e6_ddr_window_start/end`): E6 installs that window as the TOR PMP
 * entry the AME bus master uses before the application's first access. */
MEMORY { RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 4M }

SECTIONS {
  .text.nh_start 0x80000000 : { KEEP(*(.text.nh_start)) } > RAM
  .text.e6_start 0x80010000 : { KEEP(*(.text.e6_start)) } > RAM
  .text.e6 : { *(.text .text.*) } > RAM
  .rodata : { *(.rodata .rodata.* .srodata .srodata.*) } > RAM
  .data : { *(.data .data.* .sdata .sdata.*) } > RAM

  . = ALIGN(16);
  .bss (NOLOAD) : {
    __bss_start = .;
    *(.bss .bss.* .sbss .sbss.*)
    *(COMMON)
    . = ALIGN(16);
    __bss_end = .;
  } > RAM

  .stacks 0x800e0000 (NOLOAD) : {
    __nh_stack_bottom = .; . += 64K; __nh_stack_top = .;
  } > RAM

  /* A[128,128] i8 = 16384 B and B[1024,128] i8 = 131072 B share the source
   * window; C[128,1024] i32 = 524288 B is the output window.  Both fit the
   * 1 MiB they are given, and the ASSERTs below keep it that way. */
  /* NOLOAD: e6_main() fills both windows before the kernel reads them, so the
   * image does not have to carry ~660 KiB of zeros. */
  .dma_source 0x80100000 (NOLOAD) : { KEEP(*(.dma_source)) } > RAM
  .dma_output 0x80200000 (NOLOAD) : { KEEP(*(.dma_output)) } > RAM
  /* Compiler spills and CPU-written stack data live in PMP entry 6, whose
   * write-through/cache policy is stable across AME and DMA operations. */
  .e6_stack 0x802e0000 (NOLOAD) : {
    __e6_stack_bottom = .; . += 64K; __e6_stack_top = .;
  } > RAM
  . = ALIGN(4096);
  _end = .;

  __nh_e6_ddr_window_start = 0x80100000;
  __nh_e6_ddr_window_end = __e6_stack_bottom;

  /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) }
  ASSERT(_start == 0x80000000, "NH entry must be 0x80000000")
  ASSERT(e6_start == 0x80010000, "E6 entry must be 0x80010000")
  ASSERT(ADDR(.dma_source) == 0x80100000, "source window moved")
  ASSERT(ADDR(.dma_output) == 0x80200000, "output window moved")
  ASSERT(__nh_e6_ddr_window_start <= ADDR(.dma_source) &&
         ADDR(.dma_source) + SIZEOF(.dma_source) <= __nh_e6_ddr_window_end,
         "source window outside the PMP DDR window")
  ASSERT(__nh_e6_ddr_window_start <= ADDR(.dma_output) &&
         ADDR(.dma_output) + SIZEOF(.dma_output) <= __nh_e6_ddr_window_end,
         "output window outside the PMP DDR window")
  ASSERT(ADDR(.dma_source) + SIZEOF(.dma_source) <= ADDR(.dma_output),
         "source window overlaps output")
  ASSERT(ADDR(.dma_output) + SIZEOF(.dma_output) <= __e6_stack_bottom,
         "output window overlaps the E6 stack")
  ASSERT(__nh_e6_ddr_window_end <= __e6_stack_bottom,
         "E6 stack must use the stable PMP entry 6 cache policy")
  ASSERT(__bss_end <= __nh_stack_bottom, "image overlaps the NH stack")
  ASSERT(_end <= ORIGIN(RAM) + LENGTH(RAM), "image exceeds 4MiB")
}
