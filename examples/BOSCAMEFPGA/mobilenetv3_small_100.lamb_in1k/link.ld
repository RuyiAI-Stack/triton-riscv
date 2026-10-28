OUTPUT_ARCH("riscv")
ENTRY(_start)

/* NH/E6 image for one EfficientNet operator. Boot and the platform verdict
 * live in runtime/boscame/nh_e6. Operands sit in board_support.c's .workspace
 * arena (VIT_WORKSPACE_BYTES, 16 MiB on the board), which must lie inside the
 * DDR window E6 installs as PMP entry 5. The E6 stack stays after that window
 * so it keeps the stable cache policy of PMP entry 6. The arena is NOLOAD. */
MEMORY { RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 32M }
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
  .workspace 0x80100000 (NOLOAD) : {
    . = ALIGN(64);
    KEEP(*(.workspace .workspace.*))
    . = ALIGN(64);
  } > RAM
  .e6_stack 0x81100000 (NOLOAD) : {
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
  ASSERT(ADDR(.workspace) == 0x80100000, "operator workspace moved")
  ASSERT(ADDR(.workspace) + SIZEOF(.workspace) <= __e6_stack_bottom,
         "operator workspace overlaps the E6 stack")
  ASSERT(ADDR(.workspace) + SIZEOF(.workspace) <= __nh_e6_ddr_window_end,
         "operator workspace leaves the switching PMP window")
  ASSERT(__nh_e6_ddr_window_end <= __e6_stack_bottom,
         "E6 stack must use the stable PMP entry 6 cache policy")
  ASSERT(_end <= ORIGIN(RAM) + LENGTH(RAM), "image exceeds DDR")
}
