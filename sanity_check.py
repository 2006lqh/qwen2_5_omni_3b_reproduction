#!/usr/bin/env python3
"""Minimal text-only sanity check for Qwen/Qwen2.5-Omni-3B.

This script verifies that Transformers can load the model and processor, then
runs one short text prompt. It deliberately avoids tokens, datasets, training,
and multimodal downloads.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import shutil
import sys
from typing import Any


MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
REQUIRED_TRANSFORMERS = "4.52.3"


def installed_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load Qwen2.5-Omni-3B and run a minimal text inference."
    )
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help=f"Hugging Face model id. Default: {MODEL_ID}",
    )
    parser.add_argument(
        "--prompt",
        default="Please introduce yourself in one short sentence.",
        help="Text prompt for the sanity check.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum generated tokens.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Device preference. Default: auto.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only already-downloaded Hugging Face cache files.",
    )
    parser.add_argument(
        "--enable-audio-output",
        action="store_true",
        help="Keep the audio-output talker enabled. Text sanity check defaults to disabled.",
    )
    return parser.parse_args()


def print_preflight() -> None:
    transformers_version = installed_version("transformers")
    print(f"[env] transformers: {transformers_version or '<missing>'}")
    if transformers_version != REQUIRED_TRANSFORMERS:
        print(
            "[warn] Qwen2.5-Omni needs transformers=="
            f"{REQUIRED_TRANSFORMERS}; otherwise KeyError: 'qwen2_5_omni' is common."
        )

    if shutil.which("ffmpeg") is None:
        print(
            "[warn] ffmpeg was not found on PATH. Text inference can still work, "
            "but audio/video preprocessing will fail until ffmpeg is installed."
        )
    else:
        print("[env] ffmpeg: found")


def choose_device(torch_module: Any, preference: str) -> str:
    cuda_available = torch_module.cuda.is_available()
    if preference == "cuda" and not cuda_available:
        print("[warn] --device cuda was requested, but CUDA is not available. Falling back to CPU.")
        return "cpu"
    if preference == "cpu":
        return "cpu"
    if cuda_available:
        return "cuda"
    print("[warn] CUDA is not available. Falling back to CPU; loading/generation may be very slow.")
    return "cpu"


def move_inputs_to_device(inputs: Any, device: Any) -> Any:
    """Move tensors to device without changing integer token dtypes."""
    return inputs.to(device)


def explain_failure(exc: BaseException) -> None:
    message = str(exc)
    lower = message.lower()

    print("\n[error] sanity check failed.")

    if isinstance(exc, KeyError) and "qwen2_5_omni" in message:
        print(
            "- transformers does not know the qwen2_5_omni architecture. "
            "Install exactly: python -m pip install 'transformers==4.52.3'"
        )
    elif "out of memory" in lower or "cuda error" in lower and "memory" in lower:
        print(
            "- CUDA memory looks insufficient. Try closing other GPU workloads, "
            "run with --device cpu, or use a lower-memory/quantized setup later."
        )
    elif "ffmpeg" in lower:
        print("- ffmpeg appears to be missing. On Ubuntu/WSL, install it with: sudo apt install ffmpeg")
    elif "401" in lower or "403" in lower or "gated" in lower or "repo" in lower and "not found" in lower:
        print(
            "- Hugging Face model access/download failed. Check the model id, network, "
            "and whether a Hugging Face login is needed. Do not hard-code tokens in this repo."
        )
    elif "connection" in lower or "timed out" in lower or "max retries" in lower:
        print("- Network access to Hugging Face failed. Retry later or pre-download the model cache.")
    elif "no module named" in lower:
        print("- A Python dependency is missing. Activate masquant and run: python -m pip install -r requirements.txt")
    else:
        print("- See the Python exception above for details.")


def main() -> int:
    args = parse_args()
    print_preflight()

    try:
        import torch
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    except Exception as exc:
        explain_failure(exc)
        print(f"\n[exception] {type(exc).__name__}: {exc}")
        return 1

    print(f"[env] torch: {torch.__version__}")
    print(f"[env] torch cuda available: {torch.cuda.is_available()}")
    print(f"[env] torch cuda runtime: {torch.version.cuda}")

    device = choose_device(torch, args.device)
    print(f"[run] loading {args.model_id} on {device}")

    load_kwargs: dict[str, Any] = {
        "torch_dtype": "auto",
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    if device == "cuda":
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = {"": "cpu"}

    try:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(args.model_id, **load_kwargs)
        model.eval()

        if not args.enable_audio_output and hasattr(model, "disable_talker"):
            model.disable_talker()
            print("[run] audio-output talker disabled to reduce memory use.")

        processor = Qwen2_5OmniProcessor.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )

        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are Qwen, a helpful multimodal assistant. "
                            "For this check, answer with text only."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": args.prompt}],
            },
        ]

        text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = processor(text=text, return_tensors="pt", padding=True)

        target_device = next(model.parameters()).device
        inputs = move_inputs_to_device(inputs, target_device)

        print("[run] generating...")
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                return_audio=args.enable_audio_output,
            )

        text_ids = generated[0] if isinstance(generated, tuple) else generated
        prompt_len = inputs["input_ids"].shape[-1]
        new_tokens = text_ids[:, prompt_len:] if text_ids.shape[-1] > prompt_len else text_ids
        answer = processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        print("\n[ok] model loaded and generated text.")
        print("[prompt]", args.prompt)
        print("[answer]", answer or "<empty generation>")
        return 0

    except BaseException as exc:
        explain_failure(exc)
        print(f"\n[exception] {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
