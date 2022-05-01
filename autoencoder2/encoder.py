import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Flatten, Dense, InputLayer
from keras import backend as K
from tensorflow.keras import Model


class Encoder(tf.keras.Model):
    def __init__(self, args):
        #super(CVAE, self).__init__()
        super().__init__()
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
        self.build_model()

    def build_model(self):
        inputShape = (self.imageSize, self.imageSize, self.nchannels)
        filters = self.nfilters
        kernelSize = self.kernelSize
        interDim = self.interDim
        latentDim = self.latentDim

        encoderInput = InputLayer(input_shape=inputShape)

        x = encoderInput

        for i in range(self.nlayers):

            x = Conv2D(filters=filters,
                        kernel_size=kernelSize,
                        activation='relu',
                        strides=2,
                        padding='same')(x)
            filters *= 2

        shape = K.int_shape(x1)

        x = Flatten()(x)
        z_mean = Dense(latentDim)(x)
        z_log_var = Dense(latentDim)(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = Model (encoderInput, [z_mean, z_log_var, z])
        encoder.summary()