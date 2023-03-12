import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import isfile, join


def plot_comparison(configs, loss, running_loss, f1_scores, miou_scores, pred_scores):
    fig, axs = plt.subplots(1, 5, figsize=(15, 5))
    axs[0].set_title("Loss")
    axs[1].set_title("Running Loss")
    axs[2].set_title("F1 Score")
    axs[3].set_title("mIoU Score")
    axs[4].set_title("Pred Score")
    for i, config in enumerate(configs):
        axs[0].plot(loss[i], label=config)
        axs[1].plot(running_loss[i], label=config)
        axs[2].plot(f1_scores[i], label=config)
        axs[3].plot(miou_scores[i], label=config)
        axs[4].plot(pred_scores[i], label=config)
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[3].legend()
    axs[4].legend()
    plt.savefig("plots.png")


def plot_f1_scores(config_names, f1_scores):
    # Create a figure and axis object
    fig, ax = plt.subplots()

    for config, f1_score in zip(config_names, f1_scores):
        # Plot the F1 scores for class 1
        color = (np.random.random(), np.random.random(), np.random.random())
        ax.plot(
            f1_score[:, 0],
            color=color,
            label=f"Class 1 - {config}",
        )
        # Plot the F1 scores for class 2
        ax.plot(
            f1_score[:, 1],
            color=color,
            linestyle="dashed",
            label=f"Class 2 - {config}",
        )

    # Add labels and legend
    ax.set_xlabel("Epochs")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score Comparison for Different Configurations")
    ax.legend(loc="lower right")

    # Show the plot
    plt.savefig("f1_score_comparison.png")


if __name__ == "__main__":

    configs = ["Config1", "Config2"]

    # loss = np.load("metric_scores/hl_256_loss_epoch.npy")
    # running_loss = np.load("metric_scores/hl_256_iter_loss.npy")
    f1_scores = np.load("metric_scores/hl_256_f1_score.npy", allow_pickle=True)
    # miou_scores = np.load("metric_scores/hl_256_miou_score.npy", allow_pickle=True)
    # pred_scores = np.load("metric_scores/hl_256_pred_score.npy", allow_pickle=True)

    # plot_comparison(configs, loss, running_loss, f1_scores, miou_scores, pred_scores)

    f1 = np.array(
        [
            [[0.5, 0.55], [0.6, 0.65], [0.7, 0.75], [0.8, 0.85], [0.9, 0.95]],
            [[0.2, 0.25], [0.3, 0.35], [0.4, 0.45], [0.5, 0.55], [0.6, 0.65]],
        ]
    )
    print(f1_scores)

    # plot_f1_scores(configs, f1_scores)
