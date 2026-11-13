//
// uart.c for BOSCAME FPGA model 'mobilenetv3_small_100.lamb_in1k'.
//
// Inputs: buffers, headers, or board state owned by this model tree.
//
// Outputs: the symbols the case harness or model runtime expects.
//
// When used: built with the model makefile under
// examples/BOSCAMEFPGA/mobilenetv3_small_100.lamb_in1k.
//

#include "uart.h"

void init_uart(uint32_t freq, uint32_t baud) {
  // Keep the NR word-wide MMIO map and use the upstream init contract.
  uart_configure(uart_divisor_for(freq, baud));
}

void write_serial(char value) { UART_PUTC(value); }

void print_uart(const char *text) {
  while (*text != '\0')
    UART_PUTC(*text++);
}

void print_uart_int(uint32_t value) {
  static const char digits[] = "0123456789ABCDEF";
  for (int shift = 28; shift >= 0; shift -= 4)
    UART_PUTC(digits[(value >> shift) & 0xfu]);
}

void print_uart_addr(uint64_t value) {
  print_uart_int((uint32_t)(value >> 32));
  print_uart_int((uint32_t)value);
}

void print_uart_hex32(uint32_t value) {
  static const char digits[] = "0123456789abcdef";
  for (int shift = 28; shift >= 0; shift -= 4)
    UART_PUTC(digits[(value >> shift) & 0xfu]);
}

void print_uart_hex64(uint64_t value) {
  print_uart_hex32((uint32_t)(value >> 32));
  print_uart_hex32((uint32_t)value);
}

void print_uart_dec32(uint32_t value) {
  char digits[10];
  int count = 0;

  if (value == 0u) {
    UART_PUTC('0');
    return;
  }
  while (value != 0u) {
    digits[count++] = (char)('0' + (value % 10u));
    value /= 10u;
  }
  while (count != 0)
    UART_PUTC(digits[--count]);
}
