import json
from pathlib import Path
from typing import Dict, Any, Optional

import matplotlib.pyplot as plt


class Logger:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = []

    def log(self, step: int, data: Dict[str, Any]):
        row = {"step": step, **data}
        self.metrics.append(row)

    def flush(self, filename: str = "metrics.json"):
        out = self.log_dir / filename
        with out.open("w") as f:
            json.dump(self.metrics, f, indent=2)

    def plot(self, x_key: str = "step", y_key: str = "return", fname: str = "plot.png"):
        if not self.metrics:
            return
        xs = [m[x_key] for m in self.metrics if y_key in m]
        ys = [m[y_key] for m in self.metrics if y_key in m]
        plt.figure()
        plt.plot(xs, ys)
        plt.xlabel(x_key)
        plt.ylabel(y_key)
        plt.tight_layout()
        plt.savefig(self.log_dir / fname)
        plt.close()


def save_config(log_dir: Path, config: Dict[str, Any], filename: str = "config.json"):
    log_dir.mkdir(parents=True, exist_ok=True)
    # Convert Path objects to strings for JSON serialization
    serializable_config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    with (log_dir / filename).open("w") as f:
        json.dump(serializable_config, f, indent=2)

