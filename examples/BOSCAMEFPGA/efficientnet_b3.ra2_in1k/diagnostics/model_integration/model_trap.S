  .option norvc
  .section .text.efficientnet_trap,"ax",@progbits
  .align 2
  .globl efficientnet_trap_entry
  .type efficientnet_trap_entry,@function
efficientnet_trap_entry:
  /* Force write-through before publishing post-DMA trap diagnostics. */
  fence iorw, iorw
  li t0, 0x0f0f0f0f0f0f0f0f
  csrw 0xbc0, t0
  fence iorw, iorw
  csrr t1, mcause
  la t0, e6_diag_mcause
  sd t1, 0(t0)
  csrr t1, mepc
  la t0, e6_diag_mepc
  sd t1, 0(t0)
  csrr t1, mtval
  la t0, e6_diag_mtval
  sd t1, 0(t0)
  fence rw, rw
  li t0, 0x50000120
  li t1, 0xed
  sw t1, 0(t0)
  fence iorw, iorw
1: wfi
  j 1b
