"""Smoke-test a freshly exported .litertlm by running a few fixed prompts.

Tries the Python bindings (`litert_lm`) first; falls back to the
`run_litertlm` CLI binary from the LiteRT-LM runtime
(https://github.com/google-ai-edge/LiteRT-LM) if the Python API is not
installed in this environment.
"""

import argparse
import glob
import shutil
import subprocess
import sys
import time

PROMPTS = [
    "Hello! Briefly introduce yourself.",
    "Translate 'good morning' to Vietnamese.",
    "Summarize: LiteRT-LM runs LLMs on-device across Android, iOS, and desktop.",
]


def resolve_model(path: str) -> str:
    matches = glob.glob(path)
    if not matches:
        sys.exit(f"No .litertlm found at {path}")
    return matches[0]


def run_with_python_api(model_path: str) -> bool:
    try:
        from litert_lm import LlmEngine  # type: ignore
    except Exception:
        return False

    engine = LlmEngine(model_path)
    for prompt in PROMPTS:
        t0 = time.perf_counter()
        out = engine.generate(prompt, max_tokens=64)
        dt = time.perf_counter() - t0
        if not out:
            sys.exit(f"Empty generation for prompt: {prompt!r}")
        print(f"[{dt*1000:.0f} ms] {prompt!r} -> {out[:200]!r}")
    return True


def run_with_cli(model_path: str) -> None:
    binary = shutil.which("run_litertlm")
    if binary is None:
        sys.exit(
            "Neither the litert_lm Python module nor the run_litertlm CLI was "
            "found. Install LiteRT-LM "
            "(https://github.com/google-ai-edge/LiteRT-LM) or pip-install its "
            "Python bindings, then re-run this script."
        )
    for prompt in PROMPTS:
        t0 = time.perf_counter()
        result = subprocess.run(
            [binary, "--model", model_path, "--prompt", prompt,
             "--max_tokens", "64"],
            capture_output=True, text=True, check=True,
        )
        dt = time.perf_counter() - t0
        out = result.stdout.strip()
        if not out:
            sys.exit(f"Empty generation for prompt: {prompt!r}")
        print(f"[{dt*1000:.0f} ms] {prompt!r} -> {out[:200]!r}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="out/gemma3-270m-it-litertlm/*.litertlm",
                   help="Path or glob to the .litertlm bundle")
    args = p.parse_args()

    model_path = resolve_model(args.model)
    print(f"Verifying {model_path}")
    if not run_with_python_api(model_path):
        run_with_cli(model_path)
    print("OK")


if __name__ == "__main__":
    main()
