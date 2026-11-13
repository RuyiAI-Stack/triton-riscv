OUTPUT_ARCH("riscv")
ENTRY(_start)
MEMORY { RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 32M }
SECTIONS {
  .text.init 0x80000000 : { KEEP(*(.text.init)) } > RAM
  .text : { *(.text .text.*) } > RAM
  .rodata : { *(.rodata .rodata.* .srodata .srodata.*) } > RAM
  /* Weak optional trace/fence hooks create GOT entries. Keep them with the
   * loaded data; an orphan GOT after the NOLOAD workspace pads BINs by 17 MiB. */
  .data ALIGN(64) : { *(.data .data.*) *(.got .got.*) } > RAM
  .sdata ALIGN(16) : {
    __global_pointer$ = . + 0x800;
    *(.sdata .sdata.*)
  } > RAM
  .bss ALIGN(64) (NOLOAD) : {
    __bss_start = .;
    *(.bss .bss.* .sbss .sbss.*) *(COMMON)
    . = ALIGN(64); __bss_end = .;
  } > RAM
  .tdata ALIGN(64) : { *(.tdata .tdata.*) } > RAM
  .tbss ALIGN(64) (NOLOAD) : {
    __tbss_start = .; *(.tbss .tbss.*)
    . = ALIGN(64); __tbss_end = .;
  } > RAM
  .stack 0x800e0000 (NOLOAD) : {
    __stack_base = .; . += 64K; __stack_top = .;
  } > RAM
  .workspace 0x80100000 (NOLOAD) : {
    . = ALIGN(64); KEEP(*(.workspace .workspace.*))
    . = ALIGN(64);
  } > RAM
  _end = .;
  /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) }
  ASSERT(_start == 0x80000000, "NH entry moved")
  ASSERT(__tbss_end <= __stack_base, "image overlaps NH stack")
  ASSERT(__stack_top <= ADDR(.workspace), "NH stack overlaps workspace")
  ASSERT(_end <= ORIGIN(RAM) + LENGTH(RAM), "image exceeds DDR")
}
