OUTPUT_ARCH("riscv")
ENTRY(_start)
/* NH/E6 layout follows the ViT model linker. Complete output snapshots stay
 * below 4 MiB for scheduler capture. Immutable checkpoint tensors follow;
 * operands occupy the switchable DDR window and both stacks/control cells
 * remain outside it. Only control BSS is boot-cleared. */
MEMORY { RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 256M }
SECTIONS {
  .text.nh_start 0x80000000 : { KEEP(*(.text.nh_start)) } > RAM
  .text.e6_start 0x80010000 : { KEEP(*(.text.e6_start)) } > RAM
  .text.e6 : { *(.text .text.*) } > RAM
  .rodata : { *(.rodata .rodata.* .srodata .srodata.*) } > RAM
  .data : { *(.data .data.* .sdata .sdata.*) } > RAM
  .nh_stack 0x800e0000 (NOLOAD) : {
    __nh_stack_bottom = .; . += 64K; __nh_stack_top = .;
  } > RAM
  .model_exports 0x80100000 (NOLOAD) : {
    . = ALIGN(64); __model_exports_start = .;
    KEEP(*(.model_exports))
    . = ALIGN(64); __model_exports_end = .;
  } > RAM
  .model_params 0x80400000 : {
    . = ALIGN(64); KEEP(*(.model_params)) . = ALIGN(64);
  } > RAM
  .model_input : {
    . = ALIGN(64); KEEP(*(.model_input)) . = ALIGN(64);
  } > RAM
  .model_workspace (NOLOAD) : {
    . = ALIGN(64); __model_workspace_start = .;
    KEEP(*(.model_workspace))
    . = ALIGN(64); __model_workspace_end = .;
  } > RAM
  .bosc_ame_private_ddr (NOLOAD) : {
    . = ALIGN(64); __bosc_ame_private_ddr_start = .;
    . += 256K; __bosc_ame_private_ddr_end = .;
  } > RAM
  __nh_e6_ddr_window_start = 0x80100000;
  __nh_e6_ddr_window_end = __bosc_ame_private_ddr_end;
  .bss (NOLOAD) : {
    . = ALIGN(64); __bss_start = .;
    *(.bss .bss.* .sbss .sbss.*) *(COMMON)
    . = ALIGN(64); __bss_end = .;
  } > RAM
  .e6_stack (NOLOAD) : {
    . = ALIGN(64); __e6_stack_bottom = .;
    . += 2M; __e6_stack_top = .;
  } > RAM
  _end = .;
  /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) *(.riscv.attributes) }
  ASSERT(_start == 0x80000000, "NH entry moved")
  ASSERT(e6_start == 0x80010000, "E6 entry moved")
  ASSERT(ADDR(.data) + SIZEOF(.data) <= __nh_stack_bottom, "code/data overlaps NH stack")
  ASSERT(__model_exports_end <= 0x80400000, "outputs exceed near-base capture region")
  ASSERT(__bss_start >= __nh_e6_ddr_window_end, "control BSS has switchable policy")
  ASSERT(__e6_stack_bottom >= __nh_e6_ddr_window_end, "E6 stack has switchable policy")
  ASSERT(_end <= ORIGIN(RAM) + LENGTH(RAM), "model exceeds DDR allocation")
}
