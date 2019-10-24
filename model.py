from itertools import chain

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D
from keras.optimizers import Adadelta
from keras.callbacks import TensorBoard
from keras.models import Model
from keras.models import load_model
from keras import backend as K
from datetime import datetime
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


from gatherData import DataCollector
import conf


class Autoencoder:
    """
    This class handles the model creation and training, model architectures are added here.
    """
    shape = (128, 256, 1)           # dim of 32,768

    def just_scaling(self, input_img):
        x = MaxPooling2D((2, 2), padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)
        self.encoder = Model(input=input_img, output=encoded)

        x = UpSampling2D((2, 2))(encoded)
        x = UpSampling2D((2, 2))(x)
        decoded = UpSampling2D((2, 2))(x)
        self.model = Model(input_img, decoded)

    def orig_architecture(self, input_img):
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)
        self.encoder = Model(input=input_img, output=encoded)

        x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        self.model = Model(input_img, decoded)

    def some_architecture(self, input_img):
        x = Conv2D(8, (2, 2), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)
        self.encoder = Model(input=input_img, output=encoded)

        x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (2, 2), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (2, 2), activation='sigmoid', padding='same')(x)
        # decoder = Model(input=encoder.input, output=decoded)

        self.model = Model(input_img, decoded)

    def resNet_like_architecture(self, input_img):
        # Encoder
        encoder_output = Conv2D(128, (3, 3), activation='relu', padding='same', strides=1)(input_img)
        encoder_output = MaxPooling2D((2, 2), padding='same')(encoder_output)
        encoder_output = Conv2D(128, (4, 4), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(128, (3, 3), activation='relu', padding='same', strides=1)(encoder_output)
        encoder_output = MaxPooling2D((2, 2), padding='same')(encoder_output)
        encoder_output = Conv2D(256, (4, 4), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(128, (3, 3), activation='relu', padding='same', strides=1)(encoder_output)
        encoder_output = MaxPooling2D((2, 2), padding='same')(encoder_output)
        encoder_output = Conv2D(256, (4, 4), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(128, (3, 3), activation='relu', padding='same', strides=1)(encoder_output)
        encoder_output = MaxPooling2D((2, 2), padding='same')(encoder_output)
        encoder_output = Conv2D(256, (4, 4), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(128, (4, 4), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(64, (3, 3), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(16, (3, 3), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(8, (3, 3), activation='relu', padding='same')(encoder_output)
        self.encoder = Model(input=input_img, output=encoder_output)

        # Decoder
        decoder_output = Conv2D(8, (1, 1), activation='relu', padding='same')(encoder_output)
        decoder_output = Conv2D(16, (3, 3), activation='relu', padding='same')(decoder_output)
        decoder_output = Conv2D(16, (3, 3), activation='relu', padding='same')(decoder_output)
        decoder_output = Conv2D(32, (3, 3), activation='relu', padding='same')(decoder_output)
        decoder_output = Conv2D(64, (3, 3), activation='relu', padding='same')(decoder_output)
        decoder_output = UpSampling2D((2, 2))(decoder_output)
        decoder_output = Conv2D(128, (3, 3), activation='relu', padding='same')(decoder_output)
        decoder_output = UpSampling2D((2, 2))(decoder_output)
        decoder_output = Conv2D(256, (3, 3), activation='relu', padding='same')(decoder_output)
        decoder_output = Conv2D(128, (3, 3), activation='relu', padding='same')(decoder_output)
        decoder_output = UpSampling2D((2, 2))(decoder_output)
        decoder_output = Conv2D(64, (4, 4), activation='relu', padding='same')(decoder_output)
        decoder_output = Conv2D(64, (3, 3), activation='relu', padding='same')(decoder_output)
        decoder_output = Conv2D(32, (2, 2), activation='relu', padding='same')(decoder_output)
        decoder_output = Conv2D(1, (3, 3), activation='tanh', padding='same')(decoder_output)
        decoder_output = UpSampling2D((2, 2))(decoder_output)
        self.model = Model(input_img, decoder_output)

    def compile_architecture(self):
        """
        Compiles the model architecture and sets an optimizer for training.
        :return:
        """
        input_img = Input(shape=self.shape)
        self.resNet_like_architecture(input_img)

        # when using custom lr decay
        # learning_rate = 1.0
        # decay_rate = learning_rate / conf.epochs
        # adadelta = Adadelta(lr=learning_rate, rho=0.95, epsilon=1e-08, decay=decay_rate)
        # # self.model.compile(optimizer=adadelta, loss='binary_crossentropy')

        self.model.compile(optimizer='adam', loss='mean_squared_error')

        self.model.summary()
        print('Encode representation: {}'.format(self.encoder.output_shape))

    def __init__(self):
        pass

    def reshape(self, data):
        """
        Reshape data to the model input shape
        :param data:
        :return: reshaped data
        """
        return np.reshape(data, ((len(data),) + self.shape))

    def load_data(self):
        """
        Loads the needed data for training. Triggers data collection when array files are not found.
        :return: test and training data arrays
        """
        try:
            x_test = np.load(conf.dir_path + conf.test_np_file)
            x_train = np.load(conf.dir_path + conf.train_np_file)
        except FileNotFoundError:
            collector = DataCollector(conf.dir_path)
            collector.collect_data()
            collector.save_data()
            x_test = np.load(conf.dir_path + conf.test_np_file)
            x_train = np.load(conf.dir_path + conf.train_np_file)

        x_test = (librosa.power_to_db(x_test, ref=np.max) + 80) / 80
        x_test = self.reshape(x_test)

        x_train = (librosa.power_to_db(x_train, ref=np.max) + 80) / 80
        x_train = self.reshape(x_train)

        return x_test, x_train

    def get_callbacks(self):
        """
        Creates some callbacks used in training
        :return: list of callbacks
        """
        learning_rate_reduction = ReduceLROnPlateau(monitor='loss',
                                                    patience=3,
                                                    verbose=1,
                                                    factor=0.5,
                                                    min_lr=0.00001)
        checkpoint = ModelCheckpoint(conf.model_file_name,
                                     save_best_only=True,
                                     monitor='loss',
                                     mode='min')

        return [learning_rate_reduction, checkpoint]

    def train_model(self):
        """
        Trains the model and saves it to disk.
        """
        x_test, x_train = self.load_data()

        now = datetime.now().strftime('%Y%m%d-%H%M%S')

        callbacks = [TensorBoard(log_dir=conf.dir_path + '/tmp/autoencoder{}'.format(now))]
        callbacks + self.get_callbacks()

        self.model.fit(x_train, x_train,
                       epochs=conf.epochs,
                       batch_size=conf.batch_size,
                       shuffle=True,
                       validation_data=(x_test, x_test),
                       callbacks=callbacks
                       )

        now = datetime.now().strftime('%H%M%S')

        self.model.save(conf.dir_path + 'models/' + conf.model_file_name.format(time=now))
        self.encoder.save(conf.dir_path + 'models/' + 'encoder_' + conf.model_file_name.format(time=now))

    def load_model(self, model_file):
        """
        Loads an keras network model from disk.
        :param model_file:
        """
        self.model = load_model(conf.dir_path + model_file)

    def testing_model(self):
        """
        Computes a score for the model and draws some original and decoded specs for comparison.
        """
        x_test, _ = self.load_data()

        decoded_specs = self.model.predict(x_test)

        score = self.model.evaluate(x_test, x_test, batch_size=conf.batch_size, verbose=1)
        print('Test loss:', score)

        n = 10
        plt.figure(figsize=(18, 4))
        for i in range(1, n):
            # display original
            ax = plt.subplot(2, n, i)
            back_scaled = x_test[i] * 80 - 80
            librosa.display.specshow(back_scaled.reshape(128, 256),
                                     y_axis='mel',
                                     fmax=8000,
                                     x_axis='time')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + n)
            back_scaled = decoded_specs[i] * 80 - 80
            librosa.display.specshow(back_scaled.reshape(128, 256),
                                     y_axis='mel',
                                     fmax=8000,
                                     x_axis='time')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

    def test_encode(self):
        """
        Draws some feature vectors with the encoder part of the model.
        """
        x_test, _ = self.load_data()
        encoder_layer = Model(self.model.input, self.model.layers[conf.encoder_output_layer].output)
        enc_from_model = encoder_layer.predict(x_test)

        n = 10
        for i in range(1, n):
            ax = plt.subplot(1, n, i)
            plt.imshow(enc_from_model[i].reshape(16, 8*8).T)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

