eimport tensorflow as tf
from tensorflow.keras.layers import Linear, Dense
from os.path import join
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.utils import plot_model
import glob
import csv
import matplotlib.pyplot as plt

class AEGene(tf.keras.Model):
    def __init__(self, args, **kwargs):
        super(AEGene, self).__init__(**kwargs)

        self.inputDim = args.inputDim
        self.nlayers = args.nlayers
        self.nfilters = args.nfilters
        self.latentDim = args.latentDim

        self.saveDir = args.saveDir
        self.inputDir = args.inputDir
        self.epochs = args.epochs
        self.currentEpoch = 1

        self.train_loss_tracker = tf.keras.metrics.Mean(name="train_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")

        self.build_model()

    def build_model(self):
        input = Input(shape=inputShape, name = 'encoderInput')
        x = input

        x = Dense(self.latentDim, activation = 'relu')(x)

        output = Dense(self.inputDim, activation = 'relu')(x)

        self.ae = Model(input, output, name="ae")


