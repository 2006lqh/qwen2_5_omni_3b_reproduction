import os
from pathlib import Path

from lmms_eval.tasks import TaskManager


root = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
path = str(root / "benchmarks/lmms_eval_tasks")
tm = TaskManager("INFO", include_path=path, include_defaults=False)
print("\n".join(tm.all_tasks))
