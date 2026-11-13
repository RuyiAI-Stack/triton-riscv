//
// uart.h for BOSCAME FPGA model 'mobilenetv3_small_100.lamb_in1k'.
//
// Inputs: buffers, headers, or board state owned by this model tree.
//
// Outputs: the symbols the case harness or model runtime expects.
//
// When used: built with the model makefile under
// examples/BOSCAMEFPGA/mobilenetv3_small_100.lamb_in1k.
//

#ifndef __UART_H
#define __UART_H

#include <stdint.h>

// Buddy bare-metal runtime UART interface.
void init_uart(uint32_t freq, uint32_t baud);
void write_serial(char value);
void print_uart(const char *text);
void print_uart_int(uint32_t value);
void print_uart_addr(uint64_t value);
//
// Lowercase fixed-width hex and unsigned decimal text, used by the NH/E6
// report paths.  print_uart_int keeps the upstream uppercase convention.
//
void print_uart_hex32(uint32_t value);
void print_uart_hex64(uint64_t value);
void print_uart_dec32(uint32_t value);

#ifndef UART_DEFAULT_FREQ_HZ
#ifdef UART_FREQ_HZ
#define UART_DEFAULT_FREQ_HZ UART_FREQ_HZ
#else
#define UART_DEFAULT_FREQ_HZ 5529600u
#endif
#endif
#ifndef UART_DEFAULT_BAUD
#ifdef UART_BAUD
#define UART_DEFAULT_BAUD UART_BAUD
#else
#define UART_DEFAULT_BAUD 115200u
#endif
#endif

//
// The board value is fixed, but a host test may predefine UART_BASE as an
// in-memory fake register file, the same way the DMA helper lets tests
// override BOSC_AME_DMAx_BASE.
//
#ifndef UART_BASE
#define UART_BASE 0x310b0000UL
#endif
#define UART_REG(offset) (*(volatile uint32_t *)(UART_BASE + (offset)))

#define UART_RBR_THR_DLL UART_REG(0x00)
#define UART_IER_DLH UART_REG(0x04)
#define UART_FCR UART_REG(0x08)
#define UART_LCR UART_REG(0x0c)
#define UART_MCR UART_REG(0x10)
#define UART_LSR UART_REG(0x14)
#define UART_USR UART_REG(0x7c)
#define UART_THR UART_REG(0x20)

static __attribute__((unused)) uint32_t uart_divisor_for(uint32_t freq,
                                                         uint32_t baud) {
  const uint64_t denominator =
      (uint64_t)(baud ? baud : UART_DEFAULT_BAUD) * 16u;
  uint64_t divisor;

  if (freq == 0u)
    freq = UART_DEFAULT_FREQ_HZ;
  divisor = ((uint64_t)freq + denominator / 2u) / denominator;
  if (divisor < 1u)
    divisor = 1u;
  if (divisor > 0xffffu)
    divisor = 0xffffu;
  return (uint32_t)divisor;
}

static __attribute__((unused)) void uart_configure(uint32_t divisor) {
  // 关闭串口中断，此时 DLAB=0，所以 +0x04 是 IER。
  UART_IER_DLH = 0;

  //
  // DLAB=1，允许配置除数锁存器。
  // 低两位为 11，配置 8 位数据。
  //
  UART_LCR = 0x83;

  // DW APB UART USR bit0：UART Busy。
  while (UART_USR & 0x01)
    ;

  // DLAB=1 时 +0x04 是 DLH，+0x00 是 DLL。
  UART_IER_DLH = (divisor >> 8) & 0xffu;
  UART_RBR_THR_DLL = divisor & 0xffu;

  // DLAB=0，8 数据位、无校验、1 停止位，即 8N1。
  UART_LCR = 0x03;

  // 使能 FIFO，但不主动清空收发 FIFO。
  UART_FCR = 0x01;

  // 置位 DTR 和 RTS。
  UART_MCR = 0x03;
}

static __attribute__((unused)) void uart_init(void) {
  uart_configure(uart_divisor_for(UART_DEFAULT_FREQ_HZ, UART_DEFAULT_BAUD));
}

//
// Character sink hook.  It defaults to the real MMIO path; a host test can
// predefine UART_PUTC(c) before including uart.h/uart.c to capture the exact
// byte stream that the print_uart* helpers produce.
//
#ifndef UART_PUTC
static void uart_putc(char c) {
  while ((UART_LSR & 0x20) == 0)
    ;
  UART_THR = c;
}
#define UART_PUTC(c) uart_putc(c)
#endif

static __attribute__((unused)) int uart_rx_ready(void) {
  return (UART_LSR & 0x01) != 0;
}
static __attribute__((unused)) char uart_getc(void) {
  while (!uart_rx_ready())
    ;
  return (char)(UART_RBR_THR_DLL & 0xffu);
}

static __attribute__((unused)) void uart_puts(char *s) {
  while (*s != '\0') {
    UART_PUTC(*(s++));
  }
}

#endif // __UART_H
