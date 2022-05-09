import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Conv2DTranspose, Reshape
from os.path import join
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.utils import plot_model
import glob
import csv
import matplotlib.pyplot as plt


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def save_reconstruction(reconstruction, epoch):
    for image, label in reconstruction.take(len(reconstruction)):
        image = image.numpy()
        fig = plt.figure(figsize=(1, 1))
        plt.subplot(1, 1, 1)
        plt.imshow(image)
        plt.axis('off')

        # tight_layout minimizes the overlap between 2 sub-plots
        plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))

class CVAE(tf.keras.Model):
    def __init__(self, args, **kwargs):
        super(CVAE, self).__init__(**kwargs)

        self.imageSize = args.imageSize
        self.nchannel = args.nchannel
        self.nlayers = args.nlayers
        self.nfilters = args.nfilters
        self.kernelSize = args.kernelSize
        self.latentDim = args.latentDim
        self.interDim = args.interDim

        self.saveDir = args.saveDir
        self.inputDir = args.inputDir
        self.epochs = args.epochs
        self.currentEpoch = 1

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")

        self.build_model()

    def build_model(self):
        inputShape = (self.imageSize, self.imageSize, self.nchannel)

        encoderInput = Input(shape=inputShape, name = 'encoderInput')
        x = encoderInput

        filters = self.nfilters
        for i in range(self.nlayers):
            x = Conv2D(
                filters=filters,
                kernel_size=self.kernelSize,
                activation='relu',
                strides=2,
                padding='same'
            )(x)
            filters *=2

        shape = tf.keras.backend.int_shape(x)

        x = Flatten()(x)
        x = Dense(self.interDim, activation="relu")(x)
        z_mean = Dense(self.latentDim, name="z_mean")(x)
        z_log_var = Dense(self.latentDim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])

        self.encoder = Model(encoderInput, [z_mean, z_log_var, z], name="encoder")
        self.encoder.summary()
        plot_model(self.encoder, to_file=join(self.saveDir, 'encoder.png'), show_shapes=True)


        #build decoder
        latentInput = Input(shape=(self.latentDim,))
        x = Dense(shape[1] * shape[2] * shape[3], activation="relu")(latentInput)
        x = Reshape((shape[1], shape[2], shape[3]))(x)

        for i in range(self.nlayers):
            x = Conv2DTranspose(
                filters=filters,
                kernel_size=self.kernelSize,
                activation='relu',
                strides=2,
                padding='same'
            )(x)
            filters //=2

        decoderOutput = Conv2DTranspose(filters = inputShape[2], kernel_size=self.kernelSize, activation="sigmoid", padding="same")(x)

        self.decoder = Model(latentInput, decoderOutput, name="decoder")
        self.decoder.summary()
        plot_model(self.decoder, to_file=join(self.saveDir, 'decoder.png'), show_shapes=True)


    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    @tf.function
    def test_step(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        #save_reconstruction(reconstruction, self.currentEpoch)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
            )
        )
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss
        self.val_loss_tracker.update_state(total_loss)
        self.currentEpoch += 1

    def encode(self, data, list_ds, imageCount):
        print('Encoding')
        z_mean, _, _ = self.encoder.predict(data)

        fnFile = open(join(self.saveDir, 'filenames.csv'), 'w')
        with fnFile:
            writer = csv.writer(fnFile)
            for file in list_ds.take(imageCount):
                writer.writerow(file.numpy().decode("utf-8"))

        outFile = open(join(self.saveDir, 'encodings.csv'), 'w')
        with outFile:
            writer = csv.writer(outFile)
            writer.writerows(z_mean)

        print('Saved Encodings')
