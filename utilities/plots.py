import matplotlib.pyplot as plt
import numpy as np



def plot_comparison(configs, loss, accuracy, dice_scores):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].set_title('Loss')
    axs[1].set_title('Accuracy')
    axs[2].set_title('Dice Score')
    for i, config in enumerate(configs):
        axs[0].plot(loss[i], label=config)
        axs[1].plot(accuracy[i], label=config)
        axs[2].plot(dice_scores[i], label=config)
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    plt.savefig("loss_accuracy_dice_scores.png")


if __name__ == "__main__":
    # loss = np.load("loss.npy")
    # accuracy = np.load("accuracy.npy")
    # dice_scores = np.load("dice_scores.npy")

    configs = ['Config1', 'Config2', 'Config3']
    loss = [np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.2, 0.3, 0.4, 0.5]), np.array([0.3, 0.4, 0.5, 0.6])]
    accuracy = [np.array([0.8, 0.7, 0.6, 0.5]), np.array([0.7, 0.6, 0.5, 0.4]), np.array([0.6, 0.5, 0.4, 0.3])]
    dice_scores = [np.array([0.9, 0.8, 0.7, 0.6]), np.array([0.8, 0.7, 0.6, 0.5]), np.array([0.7, 0.6, 0.5, 0.4])]

    plot_comparison(configs, loss, accuracy, dice_scores)