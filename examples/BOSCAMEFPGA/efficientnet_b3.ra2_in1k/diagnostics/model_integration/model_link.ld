OUTPUT_ARCH("riscv")
ENTRY(_start)
/* Genuine params/input/reference are contiguous PROGBITS before the switchable
 * operand window. Control state and both stacks are beyond that window. */
MEMORY { RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 128M }
SECTIONS {
  .text.nh_start 0x80000000 : { KEEP(*(.text.nh_start)) } > RAM
  .text.e6_start 0x80010000 : { KEEP(*(.text.e6_start)) } > RAM
  .text.e6 : { *(.text .text.*) } > RAM
  .rodata : { *(.rodata .rodata.* .srodata .srodata.*) } > RAM
  .data : { *(.data .data.* .sdata .sdata.*) } > RAM
  /* fpga112 scheduler captures only [0x80000000,0x81000000). Dedicated
   * result storage precedes large parameters/workspace; it is written only
   * after the graph returns. Prefix snapshots occupy the same low window. */
  .model_export (NOLOAD) : {
    . = ALIGN(64);__model_export_start = .;
    KEEP(*(.model_export .model_export.*));
    . = ALIGN(64);KEEP(*(.model_checkpoint .model_checkpoint.*));
    . = ALIGN(64);__model_export_end = .;
  } > RAM
  .model_params : { . = ALIGN(64);KEEP(*(.model_params));. = ALIGN(64); } > RAM
  .model_input : { . = ALIGN(64);KEEP(*(.model_input));. = ALIGN(64); } > RAM
  .model_reference : { . = ALIGN(64);KEEP(*(.model_reference));. = ALIGN(64); } > RAM
  .model_workspace ALIGN(128K) (NOLOAD) : {
    __nh_e6_ddr_window_start = .;__model_workspace_start = .;
    KEEP(*(.model_workspace .model_workspace.*));. = ALIGN(64);__model_workspace_end = .;
  } > RAM
  .bosc_ame_private_ddr (NOLOAD) : {
    . = ALIGN(64);__bosc_ame_private_ddr_start = .;. += 256K;__bosc_ame_private_ddr_end = .;
  } > RAM
  __nh_e6_ddr_window_end = __bosc_ame_private_ddr_end;
  .bss (NOLOAD) : {
    . = ALIGN(64);__bss_start = .;*(.bss .bss.* .sbss .sbss.*) *(COMMON)
    . = ALIGN(64);__bss_end = .;
  } > RAM
  .e6_stack (NOLOAD) : { . = ALIGN(64);__e6_stack_bottom = .;. += 1M;__e6_stack_top = .; } > RAM
  .nh_stack (NOLOAD) : { . = ALIGN(64);__nh_stack_bottom = .;. += 64K;__nh_stack_top = .; } > RAM
  _end = .;
  /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) *(.eh_frame_hdr) *(.riscv.attributes) }
  ASSERT(_start == 0x80000000,"NH entry moved")
  ASSERT(e6_start == 0x80010000,"E6 entry moved")
  ASSERT(__model_export_end <= 0x81000000,"outputs exceed fpga112 16MiB capture window")
  ASSERT(__model_workspace_end <= __bosc_ame_private_ddr_start,"workspace overlaps private frames")
  ASSERT(__bosc_ame_private_ddr_end <= __nh_e6_ddr_window_end,"private frames outside PMP window")
  ASSERT(__bss_start >= __nh_e6_ddr_window_end,"control BSS has switchable cache policy")
  ASSERT(__bss_end <= __e6_stack_bottom,"control BSS overlaps E6 stack")
  ASSERT(__e6_stack_bottom >= __nh_e6_ddr_window_end,"E6 stack has switchable policy")
  ASSERT(__e6_stack_top <= __nh_stack_bottom,"CPU stacks overlap")
  ASSERT(_end <= ORIGIN(RAM)+LENGTH(RAM),"model exceeds DDR allocation")
}
