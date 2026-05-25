"""PPO training-curve plot."""

from typing import Dict

import numpy as np
import matplotlib.pyplot as plt


def plot_training_curves(log: Dict,
                         outpath: str,
                         title: str = "PPO training curves",
                         dpi: int = 120):
    """Plot accuracy, return, final entropy, and PPO losses."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    step = np.asarray(log["step"])

    axes[0, 0].plot(step, log["ep_accuracy"], color="C0", lw=1.5)
    axes[0, 0].set_title("Episode accuracy (200-ep rolling)")
    axes[0, 0].set_ylim(0, 1.02)
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_xlabel("env steps")
    axes[0, 0].set_ylabel("accuracy")

    axes[0, 1].plot(step, log["ep_return"], color="C2", lw=1.5)
    axes[0, 1].set_title("Episode return")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_xlabel("env steps")
    axes[0, 1].set_ylabel("return")

    axes[1, 0].plot(step, log["ep_final_entropy"], color="C3", lw=1.5)
    axes[1, 0].set_title("Final belief entropy")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_xlabel("env steps")
    axes[1, 0].set_ylabel("H(b_K)")

    axes[1, 1].plot(step, log["policy_loss"], label="policy", lw=1.1)
    axes[1, 1].plot(step, log["value_loss"],  label="value",  lw=1.1)
    axes[1, 1].plot(step, log["entropy"],     label="policy H", lw=1.1)
    axes[1, 1].set_title("PPO losses")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_xlabel("env steps")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
