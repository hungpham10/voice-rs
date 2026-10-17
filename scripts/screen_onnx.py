#!/usr/bin/env python3
"""Screen an ONNX model for burn-onnx compatibility BEFORE exporting/importing.

Cheap pre-flight so we don't burn hours discovering at build time that an op
cannot be lowered.

  python3 scripts/screen_onnx.py <model.onnx> [trace_frames]

Outputs the op histogram plus four red flags:
  1. ops known to be problematic for burn-onnx
  2. initializers with a 0-sized dim (burn-onnx emits `.expand([0])`)
  3. weight-norm remnants (weights must be decomposed before export)
  4. number of dynamic dims (fewer is better; only the sequence axis is ideal)
"""

import collections
import sys

import onnx
from onnx import shape_inference

# Ops that burn-onnx cannot lower at all, or that have caused hard failures.
# NOTE: `ConstantOfShape` is deliberately NOT here — RMVPE uses it and imports
# fine (verified). It only becomes fatal when its output feeds `Expand`.
FATAL = {
    "Expand",
    "RandomNormalLike",
    "RandomUniform",
    "RandomUniformToTensor",
    "ScatterND",
    "NonZero",
    "Loop",
    "If",
    "SequenceAt",
    "SequenceInsert",
    "Optional",
    "OptionalGetElement",
}

# Present but harmless on their own — reported for information only.
INFO = {"ConstantOfShape"}


def banner(title):
    print(f"\n=== {title} ===")


def screen(path, trace_frames=None):
    m = onnx.load(path)
    banner("graph")
    print("path:", path)
    print("producer:", f"{m.producer_name} {m.producer_version}")
    print("opset:", [f"{o.domain or 'ai.onnx'}:{o.version}"
                     for o in m.opset_import])

    try:
        m = shape_inference.infer_shapes(m)
    except Exception as e:  # noqa: BLE001
        print("(shape inference skipped:", e, ")")

    ops = collections.Counter(n.op_type for n in m.graph.node)
    banner("ops")
    print("total nodes:", len(m.graph.node))
    for k, v in sorted(ops.items(), key=lambda x: -x[1]):
        print(f"  {k:22s} {v}")

    problems = 0

    banner("1. ops burn-onnx cannot lower")
    hits = {op: ops[op] for op in FATAL if ops[op]}
    print(hits if hits else "none")
    if hits:
        problems += sum(hits.values())

    info = {op: ops[op] for op in INFO if ops[op]}
    if info:
        print("(present but OK on their own:",
              info, "— RMVPE has one and imports fine)")
        # only fatal when it feeds an Expand
        if not hits:
            print("    -> not feeding Expand in this graph, safe")

    banner("2. zero-sized initializers (would become .expand([0]))")
    zeros = [(i.name, list(i.dims)) for i in m.graph.initializer
             if any(d == 0 for d in i.dims)]
    print(zeros[:8] if zeros else "none")
    problems += len(zeros)

    banner("3. weight norm remnants")
    wn = sum(
        1
        for n in m.graph.node
        for a in n.attribute
        if "weight_norm" in a.name
    )
    wn_init = sum(
        1
        for i in m.graph.initializer
        if i.name.endswith("weight_g") or i.name.endswith("weight_v")
    )
    print(f"attrs={wn}  *.weight_g/weight_v initializers={wn_init}")
    problems += wn + wn_init

    banner("4. dynamic dims")
    dyn = 0
    for io_val in list(m.graph.input) + list(m.graph.output):
        for d in io_val.type.tensor_type.shape.dim:
            if d.dim_param:
                dyn += 1
    print("count:", dyn, "  (1 = ideal, only the sequence axis)")

    banner("5. parameter count")
    n_params = 0
    for i in m.graph.initializer:
        n = 1
        for d in i.dims:
            n *= d
        n_params += n
    print(f"approx {n_params / 1e6:.1f} M")

    banner("VERDICT")
    if problems:
        print(f"KHONG AN TOAN — {problems} red flag(s). Do not import;")
        print("  either fix the Python source and re-export, or pick another model.")
        return 1
    print("OK — no red flags found.")
    return 0


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    return screen(sys.argv[1])


if __name__ == "__main__":
    sys.exit(main())
