OUTPUT_ARCH("riscv")
ENTRY(_start)
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
    *(.bss .bss.* .sbss .sbss.*) *(COMMON)
    . = ALIGN(16); __bss_end = .;
  } > RAM
  .stacks 0x800e0000 (NOLOAD) : {
    __nh_stack_bottom = .; . += 64K; __nh_stack_top = .;
  } > RAM
  .dma_source 0x80100000 (NOLOAD) : { KEEP(*(.dma_source)) } > RAM
  .dma_output 0x80200000 (NOLOAD) : { KEEP(*(.dma_output)) } > RAM
  .e6_stack 0x802e0000 (NOLOAD) : {
    __e6_stack_bottom = .; . += 64K; __e6_stack_top = .;
  } > RAM
  _end = .;
  __nh_e6_ddr_window_start = 0x80100000;
  /* User DDR uses the DMA read/write policy in entry 5. Stack spills and
   * CPU-written packs use the stable write-through policy in entry 6. */
  __nh_e6_ddr_window_end = __e6_stack_bottom;
  /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) }
  ASSERT(_start == 0x80000000, "NH entry moved")
  ASSERT(e6_start == 0x80010000, "E6 entry moved")
  ASSERT(__bss_end <= __nh_stack_bottom, "image overlaps NH stack")
  ASSERT(ADDR(.dma_source) + SIZEOF(.dma_source) <= ADDR(.dma_output),
         "input and affine guards overlap output")
  ASSERT(ADDR(.dma_output) + SIZEOF(.dma_output) <= __e6_stack_bottom,
         "output and guards overlap E6 stack")
  ASSERT(ADDR(.dma_output) + SIZEOF(.dma_output) <= __nh_e6_ddr_window_end,
         "user DMA output leaves the switching PMP window")
  ASSERT(__nh_e6_ddr_window_end <= __e6_stack_bottom,
         "E6 stack must use the stable PMP entry 6 cache policy")
  ASSERT(_end <= ORIGIN(RAM) + LENGTH(RAM), "image exceeds DDR")
}
