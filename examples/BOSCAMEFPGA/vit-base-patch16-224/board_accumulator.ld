OUTPUT_ARCH("riscv")
ENTRY(_start)

/* Diagnostic only: two independent i32 reference grids, a guarded 16 MiB
 * operand arena and a managed private frame live in the known operand window.
 * The production board layouts remain unchanged. */
MEMORY { RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 32M }
SECTIONS {
  .text.nh_start 0x80000000 : { KEEP(*(.text.nh_start)) } > RAM
  .text.e6_start 0x80010000 : { KEEP(*(.text.e6_start)) } > RAM
  .text.e6 : { *(.text .text.*) } > RAM
  .rodata : { *(.rodata .rodata.* .srodata .srodata.*) } > RAM
  .data : { *(.data .data.* .sdata .sdata.*) } > RAM
  .bss (NOLOAD) : {
    . = ALIGN(64); __bss_start = .;
    *(EXCLUDE_FILE (*adapter.o) .bss .bss.* .sbss .sbss.*)
    *(COMMON)
    . = ALIGN(64); __bss_end = .;
  } > RAM
  .nh_stack 0x800e0000 (NOLOAD) : {
    __nh_stack_bottom = .; . += 64K; __nh_stack_top = .;
  } > RAM
  .vit_golden 0x80100000 : { KEEP(*(.vit_golden)) } > RAM
  ASSERT(SIZEOF(.vit_golden) <= 7M, "diagnostic golden overlaps operand arena")
  .dma_source 0x80800000 (NOLOAD) : { KEEP(*(.dma_source)) } > RAM
  .dma_scratch (NOLOAD) : {
    . = ALIGN(64);
    *adapter.o(.bss .bss.* .sbss .sbss.*)
    . = ALIGN(64);
    __dma_scratch_end = .;
  } > RAM
  .bosc_ame_private_ddr (NOLOAD) : {
    . = ALIGN(64);
    __bosc_ame_private_ddr_start = .;
    . += 256K;
    __bosc_ame_private_ddr_end = .;
  } > RAM
  .e6_stack 0x81900000 (NOLOAD) : {
    __e6_stack_bottom = .; . += 128K; __e6_stack_top = .;
  } > RAM
  _end = .;
  __nh_e6_ddr_window_start = 0x80100000;
  __nh_e6_ddr_window_end = __e6_stack_bottom;
  /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) }
  ASSERT(_start == 0x80000000, "NH entry moved")
  ASSERT(e6_start == 0x80010000, "E6 entry moved")
  ASSERT(__bss_end <= __nh_stack_bottom, "image overlaps NH stack")
  ASSERT(__bosc_ame_private_ddr_end <= __e6_stack_bottom, "private frame exceeds operand window")
  ASSERT(_end <= ORIGIN(RAM) + LENGTH(RAM), "image exceeds DDR allocation")
}
