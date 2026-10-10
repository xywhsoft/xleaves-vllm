# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the Ornith opt-in extensions from this checkout on CUDA SM70."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--cutlass-root", type=Path, required=True)
    parser.add_argument("--flash-attn-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--modules", nargs="+", help="Optional native module names to rebuild"
    )
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    output = args.build_root.resolve()
    includes = [
        args.cutlass_root.resolve() / "include",
        args.cutlass_root.resolve() / "examples/35_gemm_softmax",
        args.flash_attn_root.resolve() / "csrc/flash_attn/src",
    ]
    if not all(p.is_dir() for p in includes):
        raise ValueError(
            "CUTLASS 4.4.2 and the qualified flash-attention checkout are required"
        )
    targets = [
        ("ornith_tq_adaptive_sm70", [repo / "csrc/ornith_tq_adaptive.cu"]),
        ("ornith_tq_gqa_wmma8c_sm70", [repo / "csrc/ornith_tq_gqa_wmma8c.cu"]),
        ("ornith_tq_gqa_wmma8u_sm70", [repo / "csrc/ornith_tq_gqa_wmma8u.cu"]),
        (
            "ornith_tq_gqa_wmma8halfpre_sm70",
            [repo / "csrc/ornith_tq_gqa_wmma8halfpre.cu"],
        ),
    ]
    targets.append(
        (
            "ornith_tq_gqa_wmma8aliasvalue80_sm70",
            [repo / "csrc/ornith_tq_gqa_wmma8aliasvalue80.cu"],
        )
    )
    for variant in ["h8", "h8k4", "h8k2"]:
        stage = repo / ("csrc/attention/ornith_" + variant)
        targets.append(
            (
                "ornith_prefill_" + variant + "_sm70",
                [stage / name for name in ["bindings.cpp", "prefill.cu", "tail.cu"]],
            )
        )
    output.mkdir(parents=True, exist_ok=True)
    if args.modules:
        unknown = set(args.modules) - {name for name, _ in targets}
        if unknown:
            raise ValueError(f"Unknown native modules: {sorted(unknown)}")
        targets = [(name, sources) for name, sources in targets if name in args.modules]
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0"
    os.environ.setdefault("MAX_JOBS", "1")
    built = []
    for name, sources in targets:
        if not all(p.is_file() for p in sources):
            raise ValueError(f"Missing checked-in source for {name}")
        manifest = dict(
            module=name,
            sources=[
                dict(
                    path=str(p.relative_to(repo)),
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                )
                for p in sources
            ],
        )
        if args.dry_run:
            print(json.dumps(manifest), flush=True)
            continue
        from torch.utils.cpp_extension import load

        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        flags = ["-O3", "--use_fast_math", "-lineinfo"]
        if name.startswith("ornith_prefill_"):
            flags += ["-DFLASHATTENTION_DISABLE_BACKWARD"]
        if name == "ornith_tq_gqa_wmma8aliasvalue80_sm70":
            flags += ["--maxrregcount=80"]
        started = time.monotonic()
        module = load(
            name=name,
            sources=[str(p) for p in sources],
            build_directory=str(directory),
            extra_cflags=["-O3"],
            extra_include_paths=[str(sources[0].parent), *[str(p) for p in includes]],
            extra_cuda_cflags=flags,
            verbose=True,
        )
        manifest.update(
            cuda_cflags=flags,
            cxx_cflags=["-O3"],
            cuda_arch="7.0",
            include_paths=[str(sources[0].parent), *[str(p) for p in includes]],
            library=module.__file__,
            elapsed_s=time.monotonic() - started,
            library_sha256=hashlib.sha256(
                Path(module.__file__).read_bytes()
            ).hexdigest(),
        )
        built.append(manifest)
        print(json.dumps(manifest), flush=True)
    if not args.dry_run:
        (output / "ornith-native-manifest.json").write_text(json.dumps(built, indent=2))


if __name__ == "__main__":
    main()
