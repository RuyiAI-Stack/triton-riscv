/* NH-only full-model layout, compatible with runtime/nh/crt_uart.S. */
OUTPUT_ARCH("riscv")
ENTRY(_start)
MEMORY { RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 32M }
SECTIONS {
  .text.init 0x80000000 : { KEEP(*(.text.init)) } > RAM
  .text : { *(.text .text.*) } > RAM
  .rodata : { *(.rodata .rodata.* .srodata .srodata.*) } > RAM
  .data : { *(.data .data.*) *(.got .got.*) } > RAM
  .sdata ALIGN(16) : {
    __global_pointer$ = . + 0x800;
    *(.sdata .sdata.*)
  } > RAM
  .bss ALIGN(64) (NOLOAD) : {
    __bss_start = .;
    *(.bss .bss.* .sbss .sbss.*) *(COMMON)
    . = ALIGN(64);
    __bss_end = .;
  } > RAM
  .tdata ALIGN(64) : { *(.tdata .tdata.*) } > RAM
  .tbss ALIGN(64) (NOLOAD) : {
    __tbss_start = .;
    *(.tbss .tbss.*)
    . = ALIGN(64);
    __tbss_end = .;
  } > RAM
  .workspace 0x80100000 (NOLOAD) : {
    KEEP(*(.workspace .workspace.*))
  } > RAM
  .stack 0x80300000 (NOLOAD) : {
    __stack_base = .;
    . += 1M;
    __stack_top = .;
  } > RAM
  .model_parameters 0x80400000 : {
    KEEP(*(.model_parameters))
  } > RAM
  _end = .;
  /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) }
  ASSERT(_start == 0x80000000, "NH entry moved")
  ASSERT(__tbss_end <= 0x80100000, "code/data overlaps activation arena")
  ASSERT(ADDR(.workspace) + SIZEOF(.workspace) <= __stack_base,
         "activation arena overlaps NH stack")
  ASSERT(__stack_top <= ADDR(.model_parameters), "NH stack overlaps weights")
  ASSERT(_end <= 0x81100000, "model weights exceed reserved DDR window")
}
