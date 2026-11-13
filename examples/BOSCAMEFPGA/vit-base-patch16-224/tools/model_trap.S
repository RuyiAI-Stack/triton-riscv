  // vit model trap entry: report the real exception context even when the
  // kernel left CSR 0xbc0 in the uncached-load/write-back policy.  The stock
  // platform entry publishes the diagnostic cells without touching 0xbc0, so
  // stores can stay in the E6 cache and NH reads stale zeros.
  .option norvc
  .section .text, "ax", @progbits
  .align 2
  .globl vit_e6_trap_entry
  .type vit_e6_trap_entry, @function
vit_e6_trap_entry:
  li t0, 0x0f0f0f0f0f0f0f0f
  csrw 0xbc0, t0
  fence iorw, iorw
  csrr t2, mcause
  la t0, e6_diag_mcause
  sd t2, 0(t0)
  csrr t2, mepc
  la t0, e6_diag_mepc
  sd t2, 0(t0)
  csrr t2, mtval
  la t0, e6_diag_mtval
  sd t2, 0(t0)
  mv t2, sp
  la t0, vit_trap_sp
  sd t2, 0(t0)
  la t0, vit_kernel_case
  ld t2, 0(t0)
  la t0, vit_trap_case
  sd t2, 0(t0)
  la t0, vit_kernel_calls
  ld t2, 0(t0)
  la t0, vit_trap_calls
  sd t2, 0(t0)
  fence rw, rw
  li t0, 0x50000120
  li t1, 0xed
  sw t1, 0(t0)
  fence iorw, iorw
1:
  wfi
  j 1b
  .size vit_e6_trap_entry, . - vit_e6_trap_entry
