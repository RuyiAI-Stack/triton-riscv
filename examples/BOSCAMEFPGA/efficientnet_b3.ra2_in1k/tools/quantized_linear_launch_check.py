"""Generate native ABI/order mocks for every classifier raw stage."""

import math

from quantized_linear_pipeline import pipeline_layout, stage_cases


def mock_source(case):
    layout = pipeline_layout(case)
    stages = stage_cases(case)
    lines = [
        '#include "support.h"',
        "#include <setjmp.h>",
        "#include <signal.h>",
        "#include <sys/resource.h>",
        "#include <sys/wait.h>",
        "#include <unistd.h>",
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "typedef struct { void *allocated, *aligned; int64_t offset, size, stride; } Descriptor1D;",
        "typedef struct { int64_t rank; void *descriptor; } Unranked;",
        "static jmp_buf completed; static size_t phase, calls, total;",
        "static void require(int ok) { if (!ok) abort(); }",
    ]
    fields = {
        "QA": "qa",
        "QW": "qw",
        "RowScale": "rs",
        "WeightScale": "ws",
        "Acc": "acc",
    }
    for index, stage in enumerate(stages):
        args = [f"Unranked *u{i}" for i in range(len(stage["arguments"]))]
        args += [
            "int32_t gx",
            "int32_t gy",
            "int32_t gz",
            "int32_t x",
            "int32_t y",
            "int32_t z",
        ]
        gx, gy, gz = stage["grid"]
        lines += [
            f"void _mlir_ciface_{stage['symbol']}({', '.join(args)}) {{",
            f"  require(phase=={index} && gx=={gx} && gy=={gy} && gz=={gz});",
            f"  require(x==(int)(calls/{gy * gz}) && y==(int)((calls/{gz})%{gy}) && z==(int)(calls%{gz}));",
        ]
        for i, (arg, shape) in enumerate(
            zip(stage["arguments"], stage["argument_shapes"])
        ):
            elements = math.prod(shape)
            lines += [
                f"  require(u{i} && u{i}->rank==1 && u{i}->descriptor);",
                f"  Descriptor1D *d{i}=u{i}->descriptor;",
                f"  require(d{i}->allocated==d{i}->aligned && d{i}->offset==0 && d{i}->size=={elements} && d{i}->stride==1);",
                f"  require(d{i}->aligned && !((uintptr_t)d{i}->aligned & 63u));",
            ]
            if arg["name"] in fields:
                entry = layout[fields[arg["name"]]]
                lines.append(
                    f"  require(d{i}->aligned==workspace(VIT_WORKSPACE_BYTES-{layout['total_bytes']}+{entry['offset']}));"
                )
        lines += [
            f"  ++total; if (++calls=={math.prod(stage['grid'])}) {{ ++phase;calls=0; }}",
            "}",
        ]
    total = sum(math.prod(stage["grid"]) for stage in stages)
    lines += [
        "extern void effnet_test_adapter(MemRef2 *,MemRef2 *,MemRef1 *,MemRef2 *);",
        f"void _mlir_ciface_kernel_{case['name']}(MemRef2 *a,MemRef2 *w,MemRef1 *b,MemRef2 *o) {{",
        "  for (int test=0; test<7; ++test) {",
        "    pid_t child=fork(); require(child>=0);",
        "    if (!child) {",
        "      struct rlimit limit={0,0}; (void)setrlimit(RLIMIT_CORE,&limit);",
        "      MemRef2 bad=*a; a=&bad;",
        "      if (test==0) effnet_test_adapter(NULL,w,b,o);",
        "      if (test==1) bad.offset=-1;",
        "      if (test==2) bad.sizes[0]=0;",
        "      if (test==3) bad.strides[1]=2;",
        "      if (test==4) bad.aligned=(char *)bad.aligned+1;",
        "      if (test==5) o->aligned=bad.aligned;",
        "      if (test==6) { bad.aligned=(void *)(UINTPTR_MAX-63u);bad.offset=64; }",
        "      effnet_test_adapter(a,w,b,o); _Exit(0);",
        "    }",
        "    int status; require(waitpid(child,&status,0)==child);",
        "    require(WIFSIGNALED(status) && (WTERMSIG(status)==SIGILL || WTERMSIG(status)==SIGTRAP));",
        "  }",
        "  effnet_test_adapter(a,w,b,o);",
        f"  require(phase==5 && total=={total});",
        "  longjmp(completed,1);",
        "}",
        "int main(void) { if (!setjmp(completed)) { (void)launch();abort(); }",
        '  printf("ABI_SMOKE_PASS programs=%zu numeric_oracle_executed=false",total);return 0;}',
        "",
    ]
    return "\n".join(lines)
