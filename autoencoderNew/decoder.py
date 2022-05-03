import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Dense, Reshape, Conv2DTranspose
from tensorflow.keras import Model

class Decoder(tf.keras.Model):
    def __init__(self, args, shape):
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
        self.shape = shape


        #self.trainpath =os.path.join(self.inputDir, 'train')
        self.trainpath = self.inputDir
        self.dataSizeTrain = len(self.trainpath)

        #self.CVAE = self.build_model()
        self.build_model()

    def build_model(self):
        inputShape = (self.imageSize, self.imageSize, self.nchannels)
        filters = self.nfilters
        kernelSize = self.kernelSize
        interDim = self.interDim
        latentDim = self.latentDim
        input_shape = (self.imageSize, self.imageSize, self.nchannels)
        latent_inputs = tf.keras.Input(shape=(self.latentDim,), name='z_sampling')
        #x = Dense(self.shape[1] * self.shape[2] * self.shape[3], activation='relu')(latent_inputs)
        #x = Reshape((self.shape[1], self.shape[2], self.shape[3]))(x)
        x = Dense(128*128*128, activation='relu')(latent_inputs)
        x = Reshape((128, 128, 128))(x)

        for i in range(self.nlayers):
            x = Conv2DTranspose(filters=filters,
                                kernel_size=kernelSize,
                                activation='relu',
                                strides=2,
                                padding='same')(x)
        filters //= 2

        outputs = Conv2DTranspose(filters=input_shape[2],
                                  kernel_size=kernelSize,
                                  activation='sigmoid',
                                  padding='same',
                                  name='decoder_output')(x)

        decoder = Model(latent_inputs, outputs)
        decoder.summary()