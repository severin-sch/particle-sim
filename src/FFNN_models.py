import numpy as np
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import tensorflow.keras.models as km
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

plt.rcParams.update({'font.size': 10, 'figure.figsize': (4.7747, 4.7747)})


class FFNN:
    """Base class for feed forward neural nets

    Provides utility functions for training, testing and analysis

    :param model: a Keras model class
    :param loss: the loss function of the model
    """
    def __init__(self, model, loss):
        self.model = model

        self.train_loss = []
        self.val_loss = []
        self.loss = loss

    def reinitialize_weights(self):
        """Reinitialize weights and biases
        """
        self.model.compile(optimizer=ko.Adam(), loss=self.loss)
        self.train_loss = []
        self.val_loss = []

    def set_dataset(self, x_train, y_train, x_test, y_test):
        """Set the dataset to be trained and tested on

        :param x_train: training input
        :param y_train: training output
        :param x_test: test input
        :param y_test: test output
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test.reshape((y_test.shape[0],))
        self.reinitialize_weights()

    def train(self, batch_size=5, epochs=10, verbose=0, val_split=0.2):
        """ Fit the model on the training set

        :param batch_size: number of data points in each gradient descent step
        :param epochs: number of epochs to train for
        :param verbose: set to 2 or 3 to display training progress
        :param val_split: share of data to check validation accuracy
        """
        if epochs>1:
            history = self.model.fit(self.x_train, self.y_train,
                                     batch_size=batch_size,
                                     epochs=epochs,
                                     verbose=verbose, validation_split=val_split)
            self.train_loss += history.history['loss']
            self.val_loss += history.history['val_loss']
        else:
            l = int(self.y_train.shape[0]*(1-val_split))
            histories = Histories()
            self.model.fit(self.x_train[:l], self.y_train[:l],
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=verbose, validation_split=0,
                           validation_data=(self.x_train[l:],
                                            self.y_train[l:]),
                           callbacks=[histories])
            self.train_loss = histories.train_losses

    def predict(self, x_test=None, pred_on_train=False):
        """ Make prediction on input data

        :param x_test: set an alternative test set
        :param pred_on_train: set to true to predict on the training set
        """
        if pred_on_train:
            x_test = self.x_train
        elif x_test is None:
            x_test = self.x_test

        self.pred_on_train=pred_on_train

        y_pred = self.model.predict(x_test)
        self.y_pred = y_pred.reshape(y_pred.shape[0])


    def plot_test(self, color="black", fig=None, ax=None, density_plot=False, y_true=None, title="Standard"):
        """Plot the latest prediction

        If you set your own dataset in predict() provide a y_true set here.

        :param color: set color of predicted values
        :param fig: Matplotlib figure to plot in
        :param ax: Matplotlib axis to plot on
        :param density_plot: whether to plot the density of the predictions
        use if there are many data points in test set
        :returns: figure
        """
        if self.pred_on_train:
            y_test = self.y_train
        elif y_true is not None:
            y_test = y_true
        else:
            y_test = self.y_test
        y_pred = self.y_pred
        model = self.model
        print("MSE score of model: ", mean_squared_error(y_test, y_pred))
        print("R2 score of model: ", r2_score(y_test, y_pred))
        print(f"Normalised covariance between predicted and actual mass is {np.corrcoef(y_test, y_pred)[0, 1]}")

        if fig is None or ax is None:
            fig, ax = plt.subplots()

        indices = np.argsort(y_test)
        ax.scatter(
            y_test[indices],
            y_pred[indices],
            c=color,
            s=6,
            alpha=0.2
        )
        if density_plot:
            sns.histplot(
                x=y_test[indices],
                y=y_pred[indices],
                bins=20,
                pthresh=.1,
                cmap="mako",
                ax=ax
            )
        ax.plot(
            y_test[indices],
            y_test[indices],
            # marker="o",
            color="firebrick",
        )
        r2 = r2_score(y_test, y_pred)
        ax.legend(title=f"$R^2$: {r2:.3f}", loc="lower right")
        if title == "Standard":
            ax.set_title(f"Predictions for '{model.name}'")
        else:
            ax.set_title(title)
        ax.set_xlabel("Actual mass")
        ax.set_ylabel("Mass")
        ax.axis('equal')

        return fig

    def plot_training(self, fig=None, ax=None, title=None, legend=None, both=True):
        """ Plots training and validation loss over epochs

        :param fig: set a figure to include the plot in
        :param ax: set a position for the plot
        :param title: set title of plot
        :param legend: set legend
        :param both: plot both train and val loss, set to false if trained on 1 epoch
        :return: The figure with the plot
        """

        if fig is None or ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.train_loss)
        if both:
            ax.plot(self.val_loss)

        if title is None:
            ax.set_title(f"Training loss for {self.model.name}")
        else:
            ax.set_title(title)
        if legend is None:
            ax.legend(['train', 'validation'], loc='upper right')
        else:
            ax.legend(legend, loc='upper right')
        ax.set_ylabel('loss')
        ax.set_xlabel('batch')

        return fig

    @property
    def r2(self):
        return r2_score(self.y_test, self.y_pred)

    @property
    def mse(self):
        return mean_squared_error(self.y_test, self.y_pred)


class CNN(FFNN):
    """A convolutional neural network for regression on 2D inputs with 2 channels

    :param field_size: the size of the input field
    """
    def __init__(self, field_size=32, name="ConvNN", loss=kls.MeanSquaredError()):
        K.set_floatx('float32')

        self.cnn = km.Sequential(name=name)

        self.cnn.add(kl.Input((field_size, field_size, 2), name="Input"))
        self.cnn.add(kl.Conv2D(
            10, 3,
            name="convolutional_1", activation='linear',
            padding="same",
            data_format='channels_last')
        )
        self.cnn.add(kl.MaxPooling2D((4, 4)))
        self.cnn.add(kl.Flatten(
            data_format="channels_last")
        )
        self.cnn.add(kl.Dense(
            32, activation='sigmoid', name="Dense_2"))
        self.cnn.add(kl.Dense(
            1, activation='relu', name="Prediction"))

        self.cnn.compile(optimizer=ko.Adam(), loss=loss)

        super().__init__(self.cnn, loss=loss)


class DenseNN(FFNN):
    """ A fully connected model for regression on 2D inputs with 2 channels

    :param field_size: size of input field
    :param name: set name of model
    :param loss: set loss function as a Keras loss class
    """
    def __init__(self, field_size=32, name="DenseNN", loss=kls.MeanSquaredError()):
        Inputshape = (field_size, field_size, 2)

        K.set_floatx('float32')
        model = km.Sequential(name=name)
        model.add(kl.Flatten(input_shape=Inputshape,
                             data_format="channels_last"))
        model.add(kl.Dense(32, activation='sigmoid', name="Dense1"))
        model.add(kl.Dense(32, activation='sigmoid', name="Dense2"))
        model.add(kl.Dense(1, activation='linear', name="Prediction"))

        model.compile(optimizer=ko.Adam(), loss=loss)

        super().__init__(model, loss=loss)


class Histories(Callback):
    """Records val_loss when training for only one epoch
    set callback = [instance of histories] to record loss"""
    def on_train_begin(self, logs={}):
        self.train_losses = []

    def on_batch_end(self, batch, logs={}):
        self.train_losses.append(logs.get('loss'))

