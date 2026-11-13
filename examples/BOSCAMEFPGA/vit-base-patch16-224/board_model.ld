OUTPUT_ARCH("riscv")
ENTRY(_start)

/* Whole-model ViT-B/16 flat image.
 *
 * Layout: all PROGBITS (code, rodata, params blob, input blob, reference
 * logits) below the PMP DDR window start so the loader's contiguous image
 * covers them. Operands, workspace, heap and private frames live in the
 * switchable PMP entry 5 window. Runtime state and both CPU stacks follow
 * that window in entry 6, whose cache policy stays stable across transitions.
 * All transient storage is NOLOAD; the RAM region fits the model's real ABI.
 */
MEMORY { RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 768M }
SECTIONS {
  .text.nh_start 0x80000000 : { KEEP(*(.text.nh_start)) } > RAM
  .text.e6_start 0x80010000 : { KEEP(*(.text.e6_start)) } > RAM
  .text.e6 : { *(.text .text.*) } > RAM
  .rodata : { *(.rodata .rodata.* .srodata .srodata.*) } > RAM
  .data : { *(.data .data.* .sdata .sdata.*) } > RAM
  .model_params : {
    . = ALIGN(64);
    KEEP(*(.model_params))
    . = ALIGN(64);
  } > RAM
  .model_input : {
    . = ALIGN(64);
    KEEP(*(.model_input))
    . = ALIGN(64);
  } > RAM
  .model_reference : {
    . = ALIGN(64);
    KEEP(*(.model_reference))
    . = ALIGN(64);
  } > RAM

  .model_workspace ALIGN(128K) (NOLOAD) : {
    __nh_e6_ddr_window_start = .;
    __model_workspace_start = .;
    KEEP(*(.model_workspace .model_workspace.*))
    . = ALIGN(64);
    __model_workspace_end = .;
  } > RAM
  .model_heap (NOLOAD) : {
    . = ALIGN(1M);
    __model_heap_start = .;
    . += 256M;
    __model_heap_end = .;
  } > RAM
  .model_operand_bss (NOLOAD) : {
    . = ALIGN(64);
    __bss_start = .;
    __model_operand_bss_start = .;
    *model_adapters.o(.bss .bss.* .sbss .sbss.* COMMON)
    *adapter_*.o(.bss .bss.* .sbss .sbss.* COMMON)
    *model_launcher.o(.bss .bss.* .sbss .sbss.* COMMON)
    . = ALIGN(64);
    __model_operand_bss_end = .;
  } > RAM
  .bosc_ame_private_ddr (NOLOAD) : {
    . = ALIGN(64);
    __bosc_ame_private_ddr_start = .;
    . += 256K;
    __bosc_ame_private_ddr_end = .;
  } > RAM
  __nh_e6_ddr_window_end = __bosc_ame_private_ddr_end;

  .bss (NOLOAD) : {
    . = ALIGN(64);
    __model_control_bss_start = .;
    *(.bss .bss.* .sbss .sbss.*)
    *(COMMON)
    . = ALIGN(64);
    __model_control_bss_end = .;
    /* e6_start clears one contiguous interval: operand BSS, the unused
     * private arena, then control BSS. Both stacks remain outside it. */
    __bss_end = .;
  } > RAM
  .e6_stack (NOLOAD) : {
    . = ALIGN(16);
    __e6_stack_bottom = .;
    . += 8M;
    __e6_stack_top = .;
  } > RAM
  .nh_stack (NOLOAD) : {
    . = ALIGN(16);
    __nh_stack_bottom = .;
    . += 64K;
    __nh_stack_top = .;
  } > RAM
  _end = .;

  /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) *(.eh_frame_hdr) *(.riscv.attributes) }
  ASSERT(_start == 0x80000000, "NH entry moved")
  ASSERT(e6_start == 0x80010000, "E6 entry moved")
  ASSERT(__nh_e6_ddr_window_start >= 0x80000000, "DDR window start under image")
  ASSERT(__nh_e6_ddr_window_end > __nh_e6_ddr_window_start, "empty DDR window")
  ASSERT(__model_workspace_start == __nh_e6_ddr_window_start, "workspace outside PMP DDR window")
  ASSERT(__model_workspace_end <= __model_heap_start, "workspace overlaps heap")
  ASSERT(__model_heap_end <= __model_operand_bss_start, "heap overlaps operand BSS")
  ASSERT(__model_operand_bss_end <= __bosc_ame_private_ddr_start, "operand BSS overlaps private frames")
  ASSERT(__bosc_ame_private_ddr_end <= __nh_e6_ddr_window_end, "private frames outside PMP DDR window")
  ASSERT(__model_control_bss_start >= __nh_e6_ddr_window_end, "control BSS has switchable cache policy")
  ASSERT(__bss_end <= __e6_stack_bottom, "BSS overlaps E6 stack")
  ASSERT(__e6_stack_bottom >= __nh_e6_ddr_window_end, "E6 stack has switchable cache policy")
  ASSERT(__e6_stack_top <= __nh_stack_bottom, "E6 and NH stacks overlap")
  ASSERT(__nh_stack_bottom >= __nh_e6_ddr_window_end, "NH stack has switchable cache policy")
  ASSERT(_end <= ORIGIN(RAM) + LENGTH(RAM), "image exceeds DDR allocation")
}
