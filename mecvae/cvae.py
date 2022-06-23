import os

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Multiply
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation
from os.path import join
from tensorflow.keras import layers, Input, Model, mixed_precision
from tensorflow.keras.utils import plot_model
import csv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils import readingData, readingDataWithFileNames, createCallbacks, createDirectories


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), dtype= 'float16')
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def save_reconstruction(reconstruction, epoch):
    #tf.compat.v1.enable_eager_execution()
    #tf.config.run_functions_eagerly(True)
    #print("Test")
    #print(tf.executing_eagerly())
    if not isinstance(reconstruction, Image.Image):
        image = tf.clip_by_value(reconstruction, 0, 1)
        image = tf.cast(image, tf.float32).numpy()
        image = 255 * image
        image = image.astype(int)
        image = Image.fromarray(image)
    image.save('image_at_epoch_{:04d}.png'.format(epoch))
    #image = np.asarray(reconstruction)
    #fig = plt.figure(figsize=(1, 1))
    #plt.subplot(1, 1, 1)
    #plt.imshow(image)
    #plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    #plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))

class CVAE(tf.keras.Model):
    def __init__(self, args, **kwargs):
        super(CVAE, self).__init__(**kwargs)

        self.imageSize = args.imageSize
        self.batchSize = args.batchSize
        self.nchannel = args.nchannel
        self.nlayers = args.nlayers
        self.nfilters = args.nfilters
        self.kernelSize = args.kernelSize
        self.latentDim = args.latentDim
        self.interDim = args.interDim

        self.saveDir = args.saveDir
        self.inputDir = args.inputDir
        self.outputFilename = args.outputFilename

        self.learnRate = args.learnRate
        self.earlystop = args.earlystop
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
        mixed_precision.set_global_policy('mixed_float16')
        inputShape = (self.imageSize, self.imageSize, self.nchannel)

        encoderInput1 = Input(shape=inputShape, name = 'encoderInput1')
        x1 = encoderInput1

        filters = self.nfilters
        for i in range(self.nlayers):
            x1 = Conv2D(
                filters=filters,
                kernel_size=self.kernelSize,
                activation='relu',
                strides=2,
                padding='same'
            )(x1)
            filters *=2

        shape = tf.keras.backend.int_shape(x1)

        x1 = Flatten()(x1)
        x1 = Dense(self.interDim, activation="relu")(x1)
        z_mean1 = Dense(self.latentDim, name="z_mean1", dtype= tf.float32)(x1)
        z_log_var1 = Dense(self.latentDim, name="z_log_var1", dtype= tf.float32)(x1)
        z1 = Sampling()([z_mean1, z_log_var1])

        self.encoder1 = Model(encoderInput1, [z_mean1, z_log_var1, z1], name="encoder1")
        self.encoder1.summary()
        plot_model(self.encoder1, to_file=join(self.saveDir, 'encoder1.png'), show_shapes=True)

        #encoder2
        #transform and rotate randomly input
        encoderInput2 = Input(shape=inputShape, name = 'encoderInput2')

        x2 = encoderInput2
        x2 = RandomFlip("horizontal_and_vertical")(x2)
        #x2 = RandomRotation(factor = 0.2)(x2)



        #dataAugmentation = tf.keras.Sequential([
            #RandomFlip("horizontal_and_vertical"),
            #RandomRotation(0.2)
        #])
        #encoderInput2 = dataAugmentation(encoderInput2)



        filters = self.nfilters
        for i in range(self.nlayers):
            x2 = Conv2D(
                filters=filters,
                kernel_size=self.kernelSize,
                activation='relu',
                strides=2,
                padding='same'
            )(x2)
            filters *=2

        x2 = Flatten()(x2)
        x2 = Dense(self.interDim, activation="relu")(x2)
        z_mean2 = Dense(self.latentDim, name="z_mean2", dtype=tf.float32)(x2)
        z_log_var2 = Dense(self.latentDim, name="z_log_var2", dtype=tf.float32)(x2)
        z2 = Sampling()([z_mean2, z_log_var2])

        self.encoder2 = Model(encoderInput2, [z_mean2, z_log_var2, z2], name="encoder2")
        self.encoder2.summary()
        plot_model(self.encoder2, to_file=join(self.saveDir, 'encoder2.png'), show_shapes=True)

        z12 =Multiply()([z1,z2])


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

        decoderOutput = Conv2DTranspose(filters = inputShape[2], kernel_size=self.kernelSize, activation="sigmoid", padding="same", dtype=tf.float32)(x)

        self.decoder = Model(latentInput, decoderOutput, name="decoder")
        self.decoder.summary()
        plot_model(self.decoder, to_file=join(self.saveDir, 'decoder.png'), show_shapes=True)

        output = self.decoder(Multiply()([self.encoder1(encoderInput1)[2], self.encoder2(encoderInput2)[2]]))
        self.cvae = Model(inputs = [encoderInput1, encoderInput2], outputs = [output], name = 'cvae')

        def DecoderLoss1(data):
            z_mean1, z_log_var1, z1 = self.encoder1(data)
            reconstruction1 = self.decoder(z1)
            reconstruction_loss1 = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction1), axis=(1, 2)
                )
            )
            kl_loss1 = -0.5 * (1 + z_log_var1 - tf.square(z_mean1) - tf.exp(z_log_var1))
            kl_loss1 = tf.reduce_mean(tf.reduce_sum(kl_loss1, axis=1))
            total_loss1 = (reconstruction_loss1 + kl_loss1)/2

            return total_loss1

        def DecoderLoss2(data):
            z_mean2, z_log_var2, z2 = self.encoder2(data)
            reconstruction2 = self.decoder(z2)
            reconstruction_loss1 = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction2), axis=(1, 2)
                )
            )
            kl_loss2 = -0.5 * (1 + z_log_var2 - tf.square(z_mean2) - tf.exp(z_log_var2))
            kl_loss2 = tf.reduce_mean(tf.reduce_sum(kl_loss2, axis=1))
            total_loss2 = (reconstruction_loss2 + kl_loss2)/2

            return total_loss2

        def TotalLoss(data):
            loss1 = DecoderLoss1(data)
            loss2 = DecoderLoss2(data)

            return loss1 + loss2

        optimizer = tf.keras.optimizers.Adam(self.learnRate)
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)

        losses = {"decoder": TotalLoss}

        self.cvae.compile(loss = losses, optimizer = optimizer)
        self.cvae.summary()
        plot_model(self.cvae, to_file=join(self.saveDir, 'cvae.png'), show_shapes=True)

        #save model architectures
        self.model_dir = join(self.saveDir, 'models')
        os.makedirs(self.model_dir, exist_ok = True)
        with open(os.path.join(self.model_dir, 'arch_cvae.json'), 'w') as file:
            file.write(self.cvae.to_json())
        with open(os.path.join(self.model_dir, 'arch_encoder1.json'), 'w') as file:
            file.write(self.encoder1.to_json())
        with open(os.path.join(self.model_dir, 'arch_encoder2.json'), 'w') as file:
            file.write(self.encoder2.to_json())
        with open(os.path.join(self.model_dir, 'arch_decoder1.json'), 'w') as file:
            file.write(self.decoder.to_json())

    @tf.function(jit_compile=True)
    def train(self):
        self.saveDir = createDirectories(self.outputFilename)
        print('Created new Directory to store Output')
        train_ds, val_ds = readingData(self.inputDir,self.imageSize, self.batchSize, shuffle= True, validation = 0.001)
        print('Read in transformed images')
        callbacks = createCallbacks(self.saveDir, self.earlystop, self.nlayers, self.learnRate, self.latentDim, self.nfilters, self.kernelSize)
        print('Created callbacks')
        mixed_precision.set_global_policy('mixed_float16')


        print('Start fitting model')
        self.hist = self.cvae.fit(train_ds, epochs = self.epochs, batch_size=self.batchSize, callbacks=callbacks)

        print('saving model weights to', self.model_dir)
        self.cvaevae.save_weights(os.path.join(self.model_dir, 'weights_cvae.hdf5'))
        self.encoder1.save_weights(os.path.join(self.model_dir, 'weights_encoder1.hdf5'))
        self.encoder2.save_weights(os.path.join(self.model_dir, 'weights_encoder2.hdf5'))
        self.decoder.save_weights(os.path.join(self.model_dir, 'weights_decoder.hdf5'))

        print('Model saved. Start encoding')
        image_ds, list_ds, imageCount = readingDataWithFileNames(self.inputDir,self.imageSize)
        self.cvae.encode(image_ds, list_ds, imageCount)
        print('DONE with Training and storing Embeddings!!')


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
            z_mean1, z_log_var1, z1 = self.encoder1(data)
            reconstruction1 = self.decoder(z1)
            reconstruction_loss1 = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction1), axis=(1, 2)
                )
            )
            kl_loss1 = -0.5 * (1 + z_log_var1 - tf.square(z_mean1) - tf.exp(z_log_var1))
            kl_loss1 = tf.reduce_mean(tf.reduce_sum(kl_loss1, axis=1))
            total_loss1 = (reconstruction_loss1 + kl_loss1)/2

            z_mean2, z_log_var2, z2 = self.encoder2(data)
            reconstruction2 = self.decoder(z2)
            reconstruction_loss2 = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction2), axis=(1, 2)
                )
            )
            kl_loss2 = -0.5 * (1 + z_log_var2 - tf.square(z_mean2) - tf.exp(z_log_var2))
            kl_loss2 = tf.reduce_mean(tf.reduce_sum(kl_loss2, axis=1))
            total_loss2 = (reconstruction_loss2 + kl_loss2)/2

            total_loss = total_loss1 + total_loss2

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state((reconstruction_loss1 + reconstruction_loss2)/2)
        self.kl_loss_tracker.update_state((kl_loss1 + kl_loss2)/2)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    @tf.function
    def test_step(self, data):
        print("Start testing")
        z_mean1, z_log_var1, z1 = self.encoder1(data)
        reconstruction = self.decoder(z1)
        print("input reconstructed; Let's save it")
        print("It's saved")
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
            )
        )
        kl_loss = -0.5 * (1 + z_log_var1 - tf.square(z_mean1) - tf.exp(z_log_var1))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss
        self.val_loss_tracker.update_state(total_loss)
        self.currentEpoch += 1

    def encode(self, data, list_ds, imageCount):
        print('Encoding')
        z_mean, _, _ = self.encoder1.predict(data)

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
