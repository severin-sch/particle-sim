"""Visualises different loss functions up against eachother for the 2D domain."""

import numpy as np
import tensorflow.keras.losses as kls
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10, 'figure.dpi': 300, 'figure.figsize': (4.7747, 5)})

def plot_loss(loss_function, prediction, target, title, ax):
    loss = np.zeros(prediction.shape)
    for i in range(prediction.shape[1]):
        for j in range(prediction.shape[0]):
            loss[i,j] = loss_function(np.array([target[i,j]]), np.array([prediction[i,j]]))

    ax.set_title(title)
    ax.imshow(np.rot90(loss))
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False)
    ax.set_xlabel("Target mass")

prediction, target = np.meshgrid(np.linspace(0.1, 1, 100), np.linspace(0.1, 1, 100))

loss_functions = [lambda target, prediction: kls.huber(target, prediction, delta=0.2),
                  kls.mean_squared_error,
                  kls.mean_absolute_percentage_error]

titles = ["Huber", "MSE", "MAP"]

fig, axes = plt.subplots(1, 3, figsize=(4.7747, 2), sharey=True)

for loss_function, title, ax in zip(loss_functions, titles, axes):
    plot_loss(loss_function, prediction, target, title, ax)

axes[0].set_ylabel("Predicted mass")


plt.savefig("../latex/plots/loss_visualisation.png")
