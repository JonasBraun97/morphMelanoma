import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, InputLayer, Conv2DTranspose, BatchNormalization, Activation, Reshape
from tensorflow.keras.callbacks import TerminateOnNaN, CSVLogger, ModelCheckpoint, EarlyStopping
import os
from os import listdir
from os.path import isfile, join
from keras import backend as K

from encoder import Encoder
from decoder import Decoder

class CVAE(tf.keras.models):
    def __init__(self, args):
        super(CVAE, self).__init__()
        self.inputDir = args.inputDir
        self.outputDir = args.outputDir
        self.saveDir = args.saveDir

        #callback inputs
        self.earlystop = args.earlystop

        self.epochs = args.epochs
        self.learnRate = args.learnRate
        self.latentDim = args.latentDim
        self.nlayers = args.nlayers
        self.nfilters = args.nfilters
        self.interDim = args.interDim
        self.kernelSize = args.kernelSize
        self.batchSize = args.batchSize
        self.epsilonStd = args.epsilonStd

        self.imageSize = args.imageSize
        self.nchannels = args.nchannels


        #self.trainpath =os.path.join(self.inputDir, 'train')
        self.trainpath = self.inputDir
        self.dataSizeTrain = len(self.trainpath)
        self.files = [f for f in listdir(self.trainpath) if isfile(join(self.trainpath, f))]

        #self.CVAE = self.build_model()
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
