"""Export google/gemma-3-270m-it to a .litertlm bundle.

Thin wrapper around litert_torch.generative.export_hf. See
https://ai.google.dev/edge/litert/conversion/pytorch/genai for the full
matrix of options (NPU AOT, vision encoders, custom quantization recipes).

Usage:
    export HF_TOKEN=hf_...
    python export_gemma3_270m.py --output-dir out/gemma3-270m-it-litertlm
"""

import argparse
import os
import sys

from litert_torch.generative.export_hf import export


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="google/gemma-3-270m-it",
                   help="HF repo id or local safetensors dir")
    p.add_argument("--output-dir", default="out/gemma3-270m-it-litertlm",
                   help="Where the .litertlm bundle is written")
    p.add_argument("--quantization-recipe", default="dynamic_wi8_afp32",
                   help="Built-in recipe name or path to a recipe JSON")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.model) and not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN is not set; gemma-3-270m-it is a gated repo.")

    os.makedirs(args.output_dir, exist_ok=True)
    export.export(
        model=args.model,
        output_dir=args.output_dir,
        quantization_recipe=args.quantization_recipe,
    )
    print(f"Wrote .litertlm bundle to {args.output_dir}")


if __name__ == "__main__":
    main()
