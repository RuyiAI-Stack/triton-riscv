OUTPUT_ARCH("riscv")
ENTRY(_start)
PROVIDE(__model_export_base = 0x80100000);
/* ViT NH/E6 layout, with scheduler snapshots below its 16 MiB readback limit.
 * Only checkpoint/input/code are PROGBITS. Scratch/private frames and stacks
 * are NOLOAD; control state and stacks have a stable cache policy (PMP 6). */
MEMORY { RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 384M }
SECTIONS {
  .text.nh_start 0x80000000 : { KEEP(*(.text.nh_start)) } > RAM
  .text.e6_start 0x80010000 : { KEEP(*(.text.e6_start)) } > RAM
  .text.e6 : { *(.text .text.*) } > RAM
  .rodata : { *(.rodata .rodata.* .srodata .srodata.*) } > RAM
  .data : { *(.data .data.* .sdata .sdata.*) } > RAM
  __model_code_end = .;
  .dma_exports __model_export_base (NOLOAD) : {
    __nh_e6_ddr_window_start = .;
    KEEP(*(.dma_exports))
    __model_exports_end = .;
  } > RAM
  .model_params 0x81000000 : {
    . = ALIGN(64); KEEP(*(.model_params)) . = ALIGN(64);
  } > RAM
  .model_input : {
    . = ALIGN(64); KEEP(*(.model_input)) . = ALIGN(64);
  } > RAM
  __model_image_end = .;
  .model_workspace ALIGN(128K) (NOLOAD) : {
    __model_workspace_start = .;
    KEEP(*(.model_workspace .model_workspace.*))
    . = ALIGN(64); __model_workspace_end = .;
  } > RAM
  .model_operand_bss (NOLOAD) : {
    . = ALIGN(64); __model_operand_bss_start = .;
    *adapter_*.o(.bss .bss.* .sbss .sbss.* COMMON)
    . = ALIGN(64); __model_operand_bss_end = .;
  } > RAM
  .bosc_ame_private_ddr (NOLOAD) : {
    . = ALIGN(64); __bosc_ame_private_ddr_start = .;
    . += 256K; __bosc_ame_private_ddr_end = .;
  } > RAM
  __nh_e6_ddr_window_end = .;
  .bss (NOLOAD) : {
    . = ALIGN(64); __bss_start = .;
    *(.bss .bss.* .sbss .sbss.* COMMON)
    . = ALIGN(64); __bss_end = .;
  } > RAM
  .e6_stack (NOLOAD) : {
    . = ALIGN(16); __e6_stack_bottom = .;
    . += 8M; __e6_stack_top = .;
  } > RAM
  .nh_stack (NOLOAD) : {
    . = ALIGN(16); __nh_stack_bottom = .;
    . += 64K; __nh_stack_top = .;
  } > RAM
  _end = .;
  /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) *(.eh_frame_hdr) *(.riscv.attributes) }
  ASSERT(_start == 0x80000000, "NH entry moved")
  ASSERT(e6_start == 0x80010000, "E6 entry moved")
  ASSERT(__model_code_end <= __model_export_base, "model code overlaps exports")
  ASSERT(__model_exports_end <= 0x81000000, "exports exceed scheduler readback window")
  ASSERT(__model_image_end <= __model_workspace_start, "image overlaps workspace")
  ASSERT(__model_workspace_end <= __model_operand_bss_start, "workspace overlaps operands")
  ASSERT(__model_operand_bss_end <= __bosc_ame_private_ddr_start, "operands overlap private frames")
  ASSERT(__bss_start >= __nh_e6_ddr_window_end, "control state has switchable cache policy")
  ASSERT(__e6_stack_bottom >= __nh_e6_ddr_window_end, "E6 stack has switchable cache policy")
  ASSERT(__nh_stack_bottom >= __e6_stack_top, "CPU stacks overlap")
  ASSERT(_end <= ORIGIN(RAM) + LENGTH(RAM), "image exceeds DDR allocation")
}
