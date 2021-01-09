import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import simulation
import fields

plt.rcParams.update({'font.size': 10, 'figure.figsize': (4.7747, 4.7747)})

def simple_mass(n, rng):
    """Uniform distribution for random mass from 0.1 to 1

    :param n: number of masses
    """
    return np.random.random(n)*0.9 + 0.1


class DataBuilder:
    """Dataset of particle moving in force field, default shape: (32, 32, 2)

    :param sim_steps: number of steps to be simulated
    :param field_size: size of prediction field
    :param field_scale: multiplied by field_size to produce size of simulated field
    :param draw_mass: function returning random masses
    :param rng: chosen rng
    """
    def __init__(self, sim_steps=500, field_size=32, field_scale=8, downsize=True,
                 draw_mass=simple_mass, rng=np.random.default_rng()):

        self.pred_field_shape = (field_size, field_size)
        self.pred_field_size = field_size
        self.sim_field_size = field_size * field_scale
        self.sim_field_shape = (self.sim_field_size, self.sim_field_size)
        self.draw_mass = draw_mass

        self.rng = rng
        self.steps = sim_steps
        self.downsize = downsize

    def build(self, samples=5000, mass=None):
        """Builds the dataset
        :param samples: number of data points
        :param mass: if set to None, N random masses are drawn between .1 and 1
        :returns: x and y
        """
        if mass is None:
            mass = self.draw_mass(samples, self.rng)
        x = []
        ignored = 0
        for sample in range(samples):
            print(f"Produced {sample+1} samples\r", end="", flush=True)
            try:
                field, positions = self._simulate(self.rng, mass[sample])
                if self.downsize:
                    field = fields.downsize_field(field, self.pred_field_shape)
                    pos = simulation.four_warm_position(self.sim_field_shape,
                                                        self.pred_field_shape,
                                                        positions[-1])
                else:
                    pos = simulation.four_warm_position(self.sim_field_shape,
                                                        self.sim_field_shape,
                                                        positions[-1])

                x.append(np.stack([field, pos], axis=-1))

            except IndexError:
                # Sometimes the particle achieves incredible acceleration and
                # goes completely haywire
                print('WARNING: Ignored simulation which gave IndexError')
                ignored += 1

        print("")
        print(f"Ignored {ignored} simulations which gave IndexError")
        print(f"Size of data set: {samples-ignored}")
        return np.array(x), mass

    def _simulate(self, rng, mass, steps=500):
        """Simulate a particle in a random field

        :param rng: Numpy random number generator to use
        :param steps: Number of steps to simulate the particle for
        :return: A tuple with the field potential, and the particle position after
                 ``steps`` simulation steps.
        """

        field_potential = fields.trap(fields.scalar_field(size=self.sim_field_size, rng=rng))
        vector_field = fields.vector_field(field_potential)
        initial_pos = (self.sim_field_size // 2, self.sim_field_size // 2)
        pos = simulation.simulate(vector_field, mass, initial_pos, steps=steps)

        return field_potential, pos

    def build_n_fields(self, N, samples=2500):
        """Builds a dataset with N fields and random masses

        :param N: number of fields
        :param samples: number of data points
        :returns: input data with shape (n, field_size, field_size, 2) and output data (n, 1)
        """
        field_potential = []
        vector_fields = []
        field = []
        for i in range(N):
            print(f"Produced {i + 1} fields\r", end="", flush=True)
            field_potential.append(fields.trap(fields.scalar_field(size=self.sim_field_size, rng=self.rng)))
            vector_fields.append(fields.vector_field(field_potential[i]))
            field.append(fields.downsize_field(field_potential[i], self.pred_field_shape))
        print("")
        masses = self.draw_mass(samples, self.rng)
        x = []
        ignored = 0
        field_inds = np.zeros(samples)
        for sample in range(samples):
            print(f"Produced {sample+1} samples\r", end="", flush=True)
            if N is 1:
                j = 0
            else:
                j = np.random.randint(0, N)
            try:
                initial_pos = (self.sim_field_size // 2, self.sim_field_size // 2)
                positions = simulation.simulate(vector_fields[j], masses[sample], initial_pos, steps=self.steps)

                pos = simulation.four_warm_position(self.sim_field_shape,
                                                    self.pred_field_shape,
                                                    positions[-1])
                x.append(np.stack([field[j], pos], axis=-1))
                field_inds[sample] = j
            except IndexError:
                # Sometimes the particle achieves incredible acceleration and
                # goes completely haywire
                print('\nWARNING: Ignored simulation which gave IndexError')
                ignored += 1
        print("")
        print(f"Ignored {ignored} simulations which gave IndexError")
        print(f"Size of data set: {samples - ignored}")
        return np.array(x), masses, field_inds

    def build_alt_set(self, samples=5000, time_steps=10, sim_steps=200):
        mass = self.draw_mass(samples, self.rng)
        steps_skip = sim_steps // time_steps
        x = np.zeros((samples, 32, 32, 2))
        for sample in range(samples):
            print(f"Produced {sample+1} samples\r", end="", flush=True)

            field, positions = self._simulate(self.rng, mass[sample], sim_steps)
            field = fields.downsize_field(field, self.pred_field_shape)
            positions = positions[0::steps_skip]
            fw_positions = np.empty((time_steps, self.pred_field_size, self.pred_field_size))

            for i, pos in enumerate(positions):
                fw_positions[i] = simulation.four_warm_position(self.sim_field_shape,
                                                                self.pred_field_shape,
                                                                pos)
            sum_pos = np.sum(fw_positions[:, :, :], axis=0)
            x[sample]=np.stack([field, sum_pos], axis=-1)
        return np.array(x), mass




def final_positions_hist(x, fig=None, ax=None, title="Final positions of particles"):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    a = np.sum(x[:, :, :, 1], axis=0)
    a[a == 0] = np.NaN
    im = ax.imshow(a, origin='lower')
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    return fig


def plot_field(field=None, nine_fields=None):
    """Plot scalar fields

    :param field: set an input data point
    :param nine_fields: must be nine fields
    """
    if field is not None:
        fig, ax = plt.subplots()
        ax.imshow(field[:, :, 0], origin='lower')
        ax.set_title("A field in the training set")
    if nine_fields is not None:
        fig, axs = plt.subplots(3, 3)
        k = 0
        for i in range(3):
            for j in range(3):
                axs[i, j].imshow(nine_fields[k, :, :, 0], origin='lower')
                k += 1
        for axr in axs:
            for ax in axr:
                ax.set_axis_off()
        fig.suptitle("The first 9 fields in the dataset")


    if type(nine_fields) and type(field) is None:
        raise ValueError("Both values cannot be None")
    return fig


def MSE(true, pred):
    return np.square(np.subtract(true, pred)).mean()


def plot_training(train_loss, val_loss, model):
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title(f"Training loss for {model.name}")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def plot_test(pred_y, test_y, model, color="black", fig=None, ax=None, density_plot=False):
    """Plot the test predictions

    :param test_y: Dataset truth
    :param pred_y: Predictions by model
    :param model: Keras model doing prediction
    :param fig: Matplotlib figure to plot in
    :param ax: Matplotlib axis to plot on
    :return: The figure with the plot
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    shape = test_y.shape
    pred_y = pred_y.reshape(shape)
    indices = np.argsort(test_y)
    ax.scatter(
        test_y[indices],
        pred_y[indices],
        c=color,
        label="Predicted mass",
        s=6,
    )
    if density_plot:
        sns.histplot(
            x=test_y[indices],
            y=pred_y[indices],
            bins=20,
            pthresh=.1,
            cmap="mako",
            ax=ax
        )
    ax.plot(
        test_y[indices],
        test_y[indices],
        #marker="o",
        color="firebrick",
        label="Actual mass",
    )
    loss = MSE(test_y, pred_y)
    ax.legend(title=f"Loss: {loss}")
    ax.set_title(f"Predictions for '{model.name}'")
    ax.set_xlabel("Actual mass")
    ax.set_ylabel("Mass")
    ax.axis('equal')

    print(f"Normalised covariance between predicted and actual mass is {np.corrcoef(test_y, pred_y)[0, 1]}")
    return fig


if __name__ == "__main__":
    data = DataBuilder(100, 200, downsize=True)
    x, y, colors = data.build_n_fields(10)
    print(y)