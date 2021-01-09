import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import tensorflow.keras.models as km
from tqdm import tqdm
import matplotlib.pyplot as plt

import fields
import simulation

class CRNN(tf.keras.Model):
    def __init__(
            self,
            num_timesteps=10,
            num_fields=1,
            field_size=32,
            name="CRNN",
            spatial_model=None):
        """(Convolutional) recurrent neural network, subclassed from tf.keras.Model.

        Has two factory methods, crnn and rnn, for creating either with convolutional
        or dense layer for spatial processing.

        :param num_timesteps: Number of timesteps to prepare for in each batch.
        :param num_fields: Number of fields (in addition to particle position).
        :param field_size: Width of quadratic field.
        :param name: Name of model.
        :param spatial_model: Keras Model or Layer.

        :raises AttributeError: If spatial_model isn't provided.
        """
        if spatial_model is None:
            raise(AttributeError("spatial_model has to be supplied."))

        super().__init__(name)
        self.num_timesteps = num_timesteps
        self.num_fields = num_fields
        self.field_size = field_size

        self.spatial_input_shape = (num_timesteps, field_size, field_size, num_fields+1)
        self.rnn_input_shape = (num_timesteps, 1, int((field_size-2)//2)**2)

        self.spatial = spatial_model

        self.gru = kl.GRU(
            field_size,
            time_major=True,
            unroll=True,
            return_state=True,
            name="GRU"
        )

        self.prediction = kl.Dense(1, name="prediction", activation="linear")

    @classmethod
    def crnn(cls, name="CRNN", **kwargs):
        """Convolutional recurrent neural network, subclassed from tf.keras.Model.
        
        :param name: Name of model.
        :param **kwargs: Passed on to __init__.
        """

        spatial = km.Sequential(name='CNN')
        spatial.add(
            kl.Conv2D(
                10,
                3,
                name="convolutional",
                padding="same"
            )
        )
        spatial.add(kl.MaxPooling2D())
        spatial.add(kl.Flatten())

        return cls(**kwargs, name=name, spatial_model=spatial)

    @classmethod
    def rnn(cls, name="RNN", **kwargs):
        """Recurrent neural network, subclassed from tf.keras.Model.

        :param name: Name of model.
        :param **kwargs: Passed on to __init__.
        """
        spatial = km.Sequential(name="Dense")
        spatial.add(kl.Flatten())
        spatial.add(kl.Dense(32))

        return cls(**kwargs, name=name, spatial_model=spatial)

    def call(self, batches, **kwargs):
        """Custom call method for making batch predictions.

        :param batches: Arraylike with shape (batches, *self.spatial_input_shape).
        :param **kwargs: Ignored. Only here for compatibility.

        :return: tf.Tensor object with predictions for each batch.
        """
        num_batches = batches.shape[0] or 1
        predictions = []
        for i in range(num_batches):
            spatial = self.spatial(batches[i])
            # time_major==True => GRU expects data on format [timesteps, batches, predictors]
            gru, _ = self.gru(spatial[:, np.newaxis])
            predictions.append(self.prediction(gru))

        return tf.convert_to_tensor(predictions)

    def analyse_predictions(self, x):
        """Similar to self.call, but takes only one batch, and returns predictions from every time step.

        :param x: Arraylike with shape (*self.spatial_input_shape).

        :return: tf.Tensor object with predictions for each time step.
        """
        predictions = []
        state = None
        spatial = self.spatial(x[0])
        for t in range(spatial.shape[0]):
            gru, state = self.gru(spatial[t, np.newaxis, np.newaxis], initial_state=state)
            predictions.append(np.array(self.prediction(gru))[0,0])

        return predictions

def simple_mass(rng):
    """Simple distribution for random mass"""
    return rng.random()*0.9 + 0.1

class Agent:
    def __init__(self, loss=None, num_timesteps=20, num_fields=1, field_size=32, model_type="crnn"):
        """Agent that subclasses from base_model.BaseModel, and uses the CRNN model.

        :param loss: Subclass of tf.keras.Losses.
        :param num_timesteps: Number of time steps to simulate and send to model.
        :param num_fields: Number of fields to prepare model for.
        :param field_size: Width of quadratic field to generate and send to model.
        :param model_type: String that represents classmethod in CRNN. 'crnn' and 'rnn' valid.

        :raises: AttributeError if given model_type doesn't exist.
        """
        try:
            model = getattr(CRNN, model_type)(
                num_timesteps=num_timesteps,
                num_fields=num_fields,
                field_size=field_size
            )
        except AttributeError:
            raise AttributeError(("The model_type parameter is invalid, "
                                  "Model doesn't have a factory method of that name."))
        
        self.loss_instance = loss or kls.Huber(delta=0.2)

        model.compile(optimizer=ko.Adam(), loss=self.loss_instance)

        dummy_x = tf.zeros((1, *model.spatial_input_shape))
        dummy_y = tf.zeros(1)
        model.predict(dummy_x, dummy_y)
        
        model.summary()

        self.model = model
        self.pred_field_shape = (field_size, field_size)
        self.field_scale = 8
        self.sim_field_size = field_size * self.field_scale
        self.sim_field_shape = (self.sim_field_size, self.sim_field_size)
        self.draw_mass = simple_mass

        self.training_history = []
        self.loss = 0
        self.test_y = np.zeros(0)
        self.pred_y = np.zeros(0)

    def simulate(self, rng, steps=50):
        """Simulate a particle in a random field

        :param rng: Numpy random number generator to use
        :param steps: Number of steps to simulate the particle for
        :return: A tuple with the field potential, and the particle position after
                 ``steps`` simulation steps.
        """
        mass = self.draw_mass(rng)
        field_potential = fields.trap(fields.scalar_field(size=self.sim_field_size, rng=rng))
        vector_field = fields.vector_field(field_potential)

        initial_pos = (self.sim_field_size // 2, self.sim_field_size // 2)
        pos = simulation.simulate(vector_field, mass, initial_pos, steps=steps)

        return mass, field_potential, pos

    def _build_batch(self, samples, rng):
        skipped_steps = 10
        x, y = [], []

        ignored_samples = 0

        for sample in range(samples):
            valid_sample = False
            while not valid_sample:
                try:
                    mass, field, positions = self.simulate(rng)

                    field = np.repeat(
                        fields.downsize_field(field, self.pred_field_shape)[np.newaxis],
                        self.model.num_timesteps,
                        axis=0
                    )

                    pos = np.zeros((self.model.num_timesteps, *self.pred_field_shape))
                    for i, position in enumerate(positions[::skipped_steps]):
                        pos[i] = simulation.four_warm_position(self.sim_field_shape,
                                                            self.pred_field_shape,
                                                            position)

                    x.append(np.stack([field, pos], axis=-1))
                    y.append([mass])
                    valid_sample = True
                except IndexError:
                    # Sometimes the particle achieves incredible acceleration and
                    # goes completely haywire
                    ignored_samples += 1

        return np.array(x), np.array(y), ignored_samples

    def train(self, epochs, samples, rng=None):
        """Train the model

        :param epochs: Number of epochs to train for
        :param samples: Number of samples to create for each epoch
        :param rng: Numpy random number generator to use
        """
        if rng is None:
            rng = np.random.default_rng()

        ignored_samples = 0
        for epoch in tqdm(range(epochs), desc="Training"):
            x, y, new_ignored_samples = self._build_batch(samples, rng)
            ignored_samples += new_ignored_samples

            loss = self.model.train_on_batch(x, y)

            self.training_history.append(loss)
        print("")
        print(f"Ignored {ignored_samples} samples, because of IndexErrors.")

        print(f"\nIgnored {ignored_samples} samples, because of IndexErrors.")

    def test(self, samples, rng=None):    
        """Test the model

        :param samples: Number of samples to generate and test on
        :param rng: Numpy random number generator to use
        """
        if rng is None:
            rng = np.random.default_rng()
        
        self.test_y = np.empty(samples)
        self.pred_y = np.empty(samples)
        
        for sample in tqdm(range(samples), desc="Testing"):
            x, y, _ = self._build_batch(1, rng)
            pred_y = self.model.predict_on_batch(x)

            self.test_y[sample] = y.flatten()
            self.pred_y[sample] = pred_y.flatten()
            
        self.loss = self.model.loss(self.test_y, self.pred_y)
        self.r2 = r2_score(self.test_y, self.pred_y)
        correlation = np.corrcoef(self.test_y, self.pred_y)[0,1]
        print(f"Pearson correlation coefficient between predicted and actual data is {correlation}")
        return self.loss

    def plot_training(self, fig=None, ax=None):
        """Plot the training history

        :param fig: Matplotlib figure to plot in
        :param ax: Matplotlib axis to plot on
        :return: The figure with the plot
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        ax.set_title(f"Model training of '{self.model.name}'")
        ax.plot(self.training_history, alpha=0.8)
        ax.plot(
            np.convolve(self.training_history, np.ones(300), 'valid') / 300
        )
        ax.grid()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")

        return fig

    def plot_test(self, fig=None, ax=None):
        """Plot the test predictions

        :param fig: Matplotlib figure to plot in
        :param ax: Matplotlib axis to plot on
        :return: The figure with the plot
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        indices = np.argsort(self.test_y)
        ax.scatter(
            self.test_y[indices],
            self.pred_y[indices],
            color="black",
            s=6,
            alpha=0.1
        )
        ax.plot(
            self.test_y[indices],
            self.test_y[indices],
            #marker="o",
            color="firebrick"
        )
        ax.legend(title=f"$R^2$: {self.r2:.3}")
        ax.set_title(f"Predictions for '{self.model.name}'")
        ax.set_xlabel("Actual mass")
        ax.set_ylabel("Mass")
        ax.axis('equal')

        return fig

    def plot_predictions(self, N, fig=None, ax=None):
        """Plots predictions for each time steps against each other.

        :param N: Number of batches to generate.
        :param fig: Instance of matplotlib.pyplot.Figure.
        :param ax: Instance of matplotlib.pyplot.Axes.
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        t = np.arange(self.model.num_timesteps) + 1
        alpha = 5/N if N < 1500 else 0.005

        ax.plot([t[0], t[-1]], [1, 1], color="firebrick", label="Target")
        
        for n in tqdm(range(N)):
            x, y, _ = self._build_batch(1, rng=np.random.default_rng())
            predictions = self.model.analyse_predictions(x)
            ax.plot(t, predictions/y[0, 0], color="midnightblue", alpha=alpha)
        
        ax.plot([1], [1], color="midnightblue", label="Predictions")
        ax.axis((t[0], t[-1], 0, 5))
        ax.set_xlabel("Time steps")
        ax.set_ylabel("Scaled prediction")
        ax.legend()
        ax.set_title("Temporally pred. development")
