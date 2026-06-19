import os
from pathlib import Path


SITE_PACKAGES_RAW = os.environ.get("LMMS_EVAL_SITE_PACKAGES", "")
SITE_PACKAGES = Path(SITE_PACKAGES_RAW) if SITE_PACKAGES_RAW else None


def main():
    if SITE_PACKAGES is None:
        raise SystemExit("Set LMMS_EVAL_SITE_PACKAGES to the active environment site-packages directory.")
    templates = {
        SITE_PACKAGES / "lmms_eval/tasks/mmmu/_default_template_yaml": """metadata:\n  interleaved_format: false\n""",
    }
    for path, content in templates.items():
        if path.exists():
            print(f"exists={path}")
            continue
        path.write_text(content, encoding="utf-8")
        print(f"created={path}")


if __name__ == "__main__":
    main()
