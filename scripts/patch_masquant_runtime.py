import os
from pathlib import Path


PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
ROOT = Path(os.environ.get("MASQUANT_ROOT", PROJECT_ROOT / "third_party/EfficientAI/masquant"))


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"{label}: already patched")
        return
    if old not in text:
        raise SystemExit(f"{label}: expected text not found in {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"{label}: patched")


def main() -> None:
    main_py = ROOT / "main.py"
    lmm_omni = ROOT / "models/LMMClass_Omni.py"

    old_omni = """            kwargs = {\"device_map\": 'auto', 'enable_audio_output': False, 'attn_implementation': 'flash_attention_2'}\n            self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, **kwargs, torch_dtype=torch.float16)\n            self.model = self._model.to('cuda')\n"""
    new_omni = """            kwargs = {\"device_map\": \"auto\", \"enable_audio_output\": False, \"attn_implementation\": \"sdpa\"}\n            self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, **kwargs, torch_dtype=torch.bfloat16)\n            self.model = self._model\n"""
    replace_once(lmm_omni, old_omni, new_omni, "LMMClass_Omni sdpa")

    old_video_import = "from lmms_eval.models.model_utils.load_video import read_video_pyav_base64\n"
    new_video_import = """try:\n    from lmms_eval.models.model_utils.load_video import read_video_pyav_base64\nexcept ImportError:\n    def read_video_pyav_base64(*args, **kwargs):\n        raise ImportError(\"read_video_pyav_base64 is unavailable in this lmms_eval version; disable custom video loader.\")\n"""
    replace_once(lmm_omni, old_video_import, new_video_import, "LMMClass_Omni video import compatibility")

    old_import = "import csv\nimport pdb\n"
    new_import = "import csv\nimport pdb\nfrom lmms_eval.tasks import TaskManager\n"
    replace_once(main_py, old_import, new_import, "main.py TaskManager import")

    old_omni_eval = """            t_results = eval_multimodal.simple_evaluate(\n                vlm,\n                tasks=args.tasks_multimodal.split(\",\"),\n                num_fewshot=args.num_fewshot,\n                limit=None if args.limit_multimodal == 1.0 else args.limit_multimodal\n            )\n            results.update(t_results['results'])\n            logger.info(results)\n"""
    new_omni_eval = """            task_manager = None\n            include_path = os.getenv(\"MASQUANT_LMMS_INCLUDE_PATH\", \"\")\n            if include_path:\n                task_manager = TaskManager(\"INFO\", include_path=include_path, include_defaults=False, model_name=args.model)\n            limit_value = None if args.limit_multimodal == 1.0 else args.limit_multimodal\n            t_results = eval_multimodal.simple_evaluate(\n                vlm,\n                tasks=args.tasks_multimodal.split(\",\"),\n                num_fewshot=args.num_fewshot,\n                limit=limit_value,\n                task_manager=task_manager,\n                force_simple=True,\n            )\n            results.update(t_results['results'])\n            logger.info(results)\n            result_json = os.getenv(\"MASQUANT_LMMS_RESULT_JSON\", \"\")\n            if result_json:\n                Path(result_json).parent.mkdir(parents=True, exist_ok=True)\n                Path(result_json).write_text(json.dumps(t_results, indent=2, ensure_ascii=False) + \"\\n\", encoding=\"utf-8\")\n"""
    replace_once(main_py, old_omni_eval, new_omni_eval, "main.py omni local task manager")

    old_vl_eval = """            t_results = eval_multimodal.simple_evaluate(\n                vlm,\n                tasks=args.tasks_multimodal.split(\",\"),\n                num_fewshot=args.num_fewshot,\n                limit=None if args.limit_multimodal == 1.0 else args.limit_multimodal,\n                gen_kwargs=\"max_new_tokens=128\"\n            )\n            results.update(t_results['results'])\n            logger.info(results)\n            print(f'tasks_multimodal:  {results}')\n"""
    new_vl_eval = """            task_manager = None\n            include_path = os.getenv(\"MASQUANT_LMMS_INCLUDE_PATH\", \"\")\n            if include_path:\n                task_manager = TaskManager(\"INFO\", include_path=include_path, include_defaults=False, model_name=args.model)\n            limit_value = None if args.limit_multimodal == 1.0 else args.limit_multimodal\n            t_results = eval_multimodal.simple_evaluate(\n                vlm,\n                tasks=args.tasks_multimodal.split(\",\"),\n                num_fewshot=args.num_fewshot,\n                limit=limit_value,\n                gen_kwargs=\"max_new_tokens=128\",\n                task_manager=task_manager,\n                force_simple=True,\n            )\n            results.update(t_results['results'])\n            logger.info(results)\n            print(f'tasks_multimodal:  {results}')\n            result_json = os.getenv(\"MASQUANT_LMMS_RESULT_JSON\", \"\")\n            if result_json:\n                Path(result_json).parent.mkdir(parents=True, exist_ok=True)\n                Path(result_json).write_text(json.dumps(t_results, indent=2, ensure_ascii=False) + \"\\n\", encoding=\"utf-8\")\n"""
    replace_once(main_py, old_vl_eval, new_vl_eval, "main.py vl local task manager")


if __name__ == "__main__":
    main()
