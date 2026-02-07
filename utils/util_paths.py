import os
from datetime import datetime


def build_output_dir(configs, script_name, timestamp=None):
    task = configs.get("task", "task")
    if timestamp is None:
        timestamp = datetime.now().strftime("%m%d%H%M")
    run_dir = f"{script_name}_{timestamp}"
    return os.path.join("output", task, run_dir)


def get_output_dir(configs):
    output_dir = configs.get("output_dir")
    if output_dir:
        return output_dir
    run_name = configs.get("run_name")
    if run_name:
        return os.path.join("output", run_name)
    task = configs.get("task", "task")
    return os.path.join("output", task)
