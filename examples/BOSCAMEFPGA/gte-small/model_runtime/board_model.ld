OUTPUT_ARCH("riscv")
ENTRY(_start)
/* Like ViT's whole-model map: checkpoint PROGBITS precede the switchable
 * operand window; stacks/control remain outside that window. Low NOLOAD
 * exports keep the scheduler's physical-DDR prefix read within 16 MiB. */
MEMORY { RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 768M }
SECTIONS {
 .text.nh_start 0x80000000 : { KEEP(*(.text.nh_start)) } > RAM
 .text.e6_start 0x80010000 : { KEEP(*(.text.e6_start)) } > RAM
 .text.e6 : { *(.text .text.*) } > RAM
 .rodata : { *(.rodata .rodata.* .srodata .srodata.*) } > RAM
 .data : { *(.data .data.* .sdata .sdata.*) } > RAM
 .model_input : { . = ALIGN(64); KEEP(*(.model_input)) } > RAM
 __model_code_end = .;
 .model_exports 0x80100000 (NOLOAD) : { KEEP(*(.model_exports)) } > RAM
 .model_params 0x81000000 : { KEEP(*(.model_params)) } > RAM
 .model_workspace ALIGN(128K) (NOLOAD) : {
  __nh_e6_ddr_window_start = .;
  KEEP(*(.model_workspace))
 } > RAM
 .model_operand_bss (NOLOAD) : {
  . = ALIGN(64); __bss_start = .;
  *adapter_*.o(.bss .bss.* .sbss .sbss.* COMMON)
  . = ALIGN(64);
 } > RAM
 .bosc_ame_private_ddr (NOLOAD) : {
  . = ALIGN(64); __bosc_ame_private_ddr_start = .;
  . += 256K; __bosc_ame_private_ddr_end = .;
 } > RAM
 __nh_e6_ddr_window_end = .;
 .bss (NOLOAD) : {
  . = ALIGN(64); __model_control_bss_start = .;
  *(.bss .bss.* .sbss .sbss.*) *(COMMON)
  . = ALIGN(64); __bss_end = .;
 } > RAM
 .e6_stack (NOLOAD) : {
  . = ALIGN(64); __e6_stack_bottom = .; . += 2M; __e6_stack_top = .;
 } > RAM
 .nh_stack (NOLOAD) : {
  . = ALIGN(64); __nh_stack_bottom = .; . += 64K; __nh_stack_top = .;
 } > RAM
 _end = .;
 /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) *(.riscv.attributes) }
 ASSERT(_start == 0x80000000, "NH entry moved")
 ASSERT(e6_start == 0x80010000, "E6 entry moved")
 ASSERT(__model_code_end <= 0x80100000, "code/input overlaps exports")
 ASSERT(SIZEOF(.model_exports) <= 15M, "model outputs exceed low export arena")
 ASSERT(__bosc_ame_private_ddr_end <= __nh_e6_ddr_window_end, "private frames outside PMP")
 ASSERT(__model_control_bss_start >= __nh_e6_ddr_window_end, "unstable control cache policy")
 ASSERT(__e6_stack_bottom >= __nh_e6_ddr_window_end, "unstable E6 stack cache policy")
 ASSERT(_end <= ORIGIN(RAM) + LENGTH(RAM), "model exceeds RAM allocation")
}
