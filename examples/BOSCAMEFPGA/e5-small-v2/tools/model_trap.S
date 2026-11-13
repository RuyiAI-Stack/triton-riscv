  .option norvc
  .section .text,"ax",@progbits
  .align 2
  .globl e5_model_trap_entry
e5_model_trap_entry:
  li t0,0x0f0f0f0f0f0f0f0f
  csrw 0xbc0,t0
  fence iorw,iorw
  csrr t1,mcause
  la t0,e6_diag_mcause
  sd t1,0(t0)
  csrr t1,mepc
  la t0,e6_diag_mepc
  sd t1,0(t0)
  csrr t1,mtval
  la t0,e6_diag_mtval
  sd t1,0(t0)
  fence rw,rw
  li t0,0x50000120
  li t1,0xed
  sw t1,0(t0)
  fence iorw,iorw
1:
  wfi
  j 1b
