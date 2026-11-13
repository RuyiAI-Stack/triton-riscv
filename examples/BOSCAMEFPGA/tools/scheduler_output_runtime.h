//
// scheduler_output_runtime.h for BOSCAME FPGA model 'tools'.
//
// Inputs: buffers, headers, or board state owned by this model tree.
//
// Outputs: the symbols the case harness or model runtime expects.
//
// When used: built with the model makefile under examples/BOSCAMEFPGA/tools.
//

#ifndef BOSCAME_SCHEDULER_OUTPUT_RUNTIME_H
#define BOSCAME_SCHEDULER_OUTPUT_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

// Initializer order is shared with scheduler_outputs.output_header().
typedef struct {
  const char *name;
  const char *dtype;
  unsigned rank;
  uint64_t shape[4];
  size_t nbytes;
  size_t offset;
} VitBoardOutputMetadata;

// E6: capture each generated slot exactly once, before its source is reused.
// The fixed metadata determines the required byte length and snapshot offset.
// Invalid slots, duplicate captures, and invalid metadata trap.
//
size_t vit_board_output_bytes(unsigned slot);
void vit_board_capture_output(unsigned slot, const void *data, size_t bytes);

// E6: return 0 after all snapshots are complete and published; -1 otherwise.
// NH: readiness is a boolean (1 only after successful finish), read after DONE.
// NH: export returns 0 on success, -1 before DONE/PASS or if not ready. It only
// emits metadata; scheduler later reads the unchanged physical DDR snapshots.
//
int vit_board_finish_outputs(void);
unsigned vit_board_outputs_ready(void);
int vit_board_export_outputs(void);

#endif
