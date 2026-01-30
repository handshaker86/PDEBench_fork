"""Training loss logger: persist and restore loss history across runs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class TrainingLogger:
    """Logs train/val loss and epoch time; saves to JSON and plots curves."""

    def __init__(self, save_path: str, model_name: str):
        self.save_path = Path(save_path)
        self.model_name = model_name
        self.loss_history_file = self.save_path / f"{model_name}_loss_history.json"

        self.epochs: List[int] = []
        self.train_loss_step: List[float] = []
        self.train_loss_full: List[float] = []
        self.val_loss_step: List[float] = []
        self.val_loss_full: List[float] = []
        self.epoch_times: List[Optional[float]] = []
        self.total_times: List[Optional[float]] = []

    def load_history(self) -> bool:
        """Load loss history from JSON; returns True if loaded."""
        if self.loss_history_file.exists():
            try:
                with open(self.loss_history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                self.epochs = data.get("epochs", [])
                self.train_loss_step = data.get("train_loss_step", [])
                self.train_loss_full = data.get("train_loss_full", [])
                self.val_loss_step = data.get("val_loss_step", [])
                self.val_loss_full = data.get("val_loss_full", [])
                self.epoch_times = data.get("epoch_times", [])
                self.total_times = data.get("total_times", [])

                logger.info(f"Loaded loss history: {len(self.epochs)} epochs")
                return True
            except Exception as e:
                logger.warning(f"Failed to load loss history: {e}, starting fresh")
                return False
        return False

    def record(
        self,
        epoch: int,
        train_loss_step: float,
        train_loss_full: float,
        val_loss_step: Optional[float] = None,
        val_loss_full: Optional[float] = None,
        epoch_time: Optional[float] = None,
    ):
        """Append one epoch's losses and optional epoch_time (seconds)."""
        self.epochs.append(epoch)
        self.train_loss_step.append(train_loss_step)
        self.train_loss_full.append(train_loss_full)

        if val_loss_step is not None:
            self.val_loss_step.append(val_loss_step)
        elif len(self.val_loss_step) < len(self.epochs):
            self.val_loss_step.append(None)

        if val_loss_full is not None:
            self.val_loss_full.append(val_loss_full)
        elif len(self.val_loss_full) < len(self.epochs):
            self.val_loss_full.append(None)

        if epoch_time is not None:
            self.epoch_times.append(epoch_time)
            total_time = (
                self.total_times[-1] + epoch_time if self.total_times else epoch_time
            )
            self.total_times.append(total_time)
        elif len(self.epoch_times) < len(self.epochs):
            self.epoch_times.append(None)
            self.total_times.append(None)

    def save(self):
        """Write loss history to JSON (atomic write via temp file)."""
        self.save_path.mkdir(parents=True, exist_ok=True)

        data = {
            "epochs": self.epochs,
            "train_loss_step": self.train_loss_step,
            "train_loss_full": self.train_loss_full,
            "val_loss_step": self.val_loss_step,
            "val_loss_full": self.val_loss_full,
            "epoch_times": self.epoch_times,
            "total_times": self.total_times,
        }

        try:
            temp_file = self.loss_history_file.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            temp_file.replace(self.loss_history_file)
            logger.debug(f"Loss history saved to {self.loss_history_file}")
        except Exception as e:
            logger.error(f"Failed to save loss history: {e}")

    def plot_loss_curves(self, save_plot: bool = True):
        """Plot train/val loss (and total time if present); optionally save PNG."""
        if len(self.epochs) == 0:
            logger.warning("No loss data to plot")
            return

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        epochs_array = np.array(self.epochs)

        axes[0].plot(
            epochs_array,
            self.train_loss_step,
            label="Train Loss (Step)",
            marker="o",
            markersize=3,
        )
        axes[0].plot(
            epochs_array,
            self.train_loss_full,
            label="Train Loss (Full)",
            marker="s",
            markersize=3,
        )
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss", color="black")
        axes[0].set_title(f"{self.model_name} - Training Loss")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale("log")
        axes[0].tick_params(axis="y", labelcolor="black")

        if self.total_times and any(t is not None for t in self.total_times):
            time_epochs = [
                e for e, t in zip(self.epochs, self.total_times) if t is not None
            ]
            time_values = [t for t in self.total_times if t is not None]

            if time_epochs:
                ax1_twin = axes[0].twinx()
                ax1_twin.plot(
                    time_epochs,
                    time_values,
                    label="Total Training Time",
                    color="red",
                    linestyle="--",
                    marker="^",
                    markersize=3,
                )
                ax1_twin.set_ylabel("Total Time (seconds)", color="red")
                ax1_twin.tick_params(axis="y", labelcolor="red")
                ax1_twin.legend(loc="upper right")

        if any(v is not None for v in self.val_loss_step):
            val_epochs = [
                e for e, v in zip(self.epochs, self.val_loss_step) if v is not None
            ]
            val_step = [v for v in self.val_loss_step if v is not None]
            val_full = [v for v in self.val_loss_full if v is not None]

            if val_epochs:
                axes[1].plot(
                    val_epochs,
                    val_step,
                    label="Val Loss (Step)",
                    marker="o",
                    markersize=3,
                )
                if any(v is not None for v in val_full):
                    axes[1].plot(
                        val_epochs,
                        val_full,
                        label="Val Loss (Full)",
                        marker="s",
                        markersize=3,
                    )
                axes[1].set_xlabel("Epoch")
                axes[1].set_ylabel("Loss")
                axes[1].set_title(f"{self.model_name} - Validation Loss")
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                axes[1].set_yscale("log")

        plt.tight_layout()

        if save_plot:
            plot_path = self.save_path / f"{self.model_name}_loss_curves.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            logger.info(f"Loss curves saved to {plot_path}")
        else:
            plt.show()

        plt.close(fig)

    def get_last_epoch(self) -> int:
        """Return last recorded epoch index, or -1 if empty."""
        if self.epochs:
            return self.epochs[-1]
        return -1
