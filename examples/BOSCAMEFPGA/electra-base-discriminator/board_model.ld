OUTPUT_ARCH("riscv")
ENTRY(_start)
/* Model operands use switchable PMP entry 5. Control and CPU stacks stay
 * outside it. Snapshots are within the scheduler's 16 MiB prefix window. */
MEMORY { RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 256M }
SECTIONS {
  .text.nh_start 0x80000000 : { KEEP(*(.text.nh_start)) } > RAM
  .text.e6_start 0x80010000 : { KEEP(*(.text.e6_start)) } > RAM
  .text.e6 : { *(.text .text.*) } > RAM
  .rodata : { *(.rodata .rodata.* .srodata .srodata.*) } > RAM
  .data : { *(.data .data.* .sdata .sdata.*) } > RAM
  .bss (NOLOAD) : {
    . = ALIGN(64); __bss_start = .;
    *(.bss .bss.* .sbss .sbss.* COMMON)
    . = ALIGN(64); __bss_end = .;
  } > RAM
  .nh_stack 0x800e0000 (NOLOAD) : {
    __nh_stack_bottom = .; . += 64K; __nh_stack_top = .;
  } > RAM
  .dma_exports 0x80100000 (NOLOAD) : {
    KEEP(*(.dma_exports))
    __model_exports_end = .;
  } > RAM
  .model_input 0x80700000 : { KEEP(*(.model_input)) } > RAM
  .model_params 0x80800000 : {
    KEEP(*(.model_params))
    __model_params_end = .;
  } > RAM
  .model_workspace 0x8d000000 (NOLOAD) : {
    KEEP(*(.model_workspace))
    __model_workspace_end = .;
  } > RAM
  .bosc_ame_private_ddr (NOLOAD) : {
    . = ALIGN(64); __bosc_ame_private_ddr_start = .;
    . += 256K; __bosc_ame_private_ddr_end = .;
  } > RAM
  .e6_stack 0x8f000000 (NOLOAD) : {
    __e6_stack_bottom = .; . += 2M; __e6_stack_top = .;
  } > RAM
  _end = .;
  __nh_e6_ddr_window_start = 0x80100000;
  __nh_e6_ddr_window_end = __e6_stack_bottom;
  /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) *(.riscv.attributes) }
  ASSERT(_start == 0x80000000, "NH entry moved")
  ASSERT(e6_start == 0x80010000, "E6 entry moved")
  ASSERT(__bss_end <= __nh_stack_bottom, "code/control overlaps NH stack")
  ASSERT(__model_exports_end <= 0x80700000, "snapshots exceed reserved capture area")
  ASSERT(__model_params_end <= 0x8d000000, "checkpoint overlaps workspace")
  ASSERT(__model_workspace_end <= __bosc_ame_private_ddr_start, "workspace overlaps private frames")
  ASSERT(__bosc_ame_private_ddr_end <= __e6_stack_bottom, "private frames overlap E6 stack")
  ASSERT(_end <= ORIGIN(RAM) + LENGTH(RAM), "model exceeds allocated DDR")
}
