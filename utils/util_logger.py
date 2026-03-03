import atexit
import os
import sys
from datetime import datetime


class _StreamTee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


class TrainLogger:
    def __init__(self, log_path, log_file, stdout_original, stderr_original, capture_stderr):
        self.log_path = log_path
        self._log_file = log_file
        self._stdout_original = stdout_original
        self._stderr_original = stderr_original
        self._capture_stderr = capture_stderr
        self._closed = False

    def close(self):
        if self._closed:
            return
        sys.stdout = self._stdout_original
        if self._capture_stderr:
            sys.stderr = self._stderr_original
        self._log_file.flush()
        self._log_file.close()
        self._closed = True


_def_output_relative_path = "output"


def _extract_run_suffix(output_dir):
    normalized = os.path.normpath(output_dir)
    parts = normalized.split(os.sep)

    if _def_output_relative_path in parts:
        output_index = parts.index(_def_output_relative_path)
        suffix_parts = parts[output_index + 1 :]
        if suffix_parts:
            return os.path.join(*suffix_parts)
        return ""

    output_prefix = _def_output_relative_path + os.sep
    if normalized.startswith(output_prefix):
        return normalized[len(output_prefix) :]

    return os.path.basename(normalized)


def setup_train_logger(output_dir, filename="train_output.log", capture_stderr=False):
    run_suffix = _extract_run_suffix(output_dir)
    log_dir = os.path.join("logs", run_suffix) if run_suffix else "logs"
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, filename)
    log_file = open(log_path, "a", encoding="utf-8")
    log_file.write(f"\n===== Train Session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    log_file.flush()

    stdout_original = sys.stdout
    stderr_original = sys.stderr

    sys.stdout = _StreamTee(stdout_original, log_file)
    if capture_stderr:
        sys.stderr = _StreamTee(stderr_original, log_file)

    logger = TrainLogger(log_path, log_file, stdout_original, stderr_original, capture_stderr)
    atexit.register(logger.close)
    return logger
