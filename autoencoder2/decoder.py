import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Dense, Reshape, Conv2DTranspose
from encoder import Encoder

class Decoder(tf.keras.Model):
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
        self.shape = Encoder.shape


        #self.trainpath =os.path.join(self.inputDir, 'train')
        self.trainpath = self.inputDir
        self.dataSizeTrain = len(self.trainpath)

        #self.CVAE = self.build_model()
        self.build_model()

    def build_model(self):
        input_shape = (self.imageSize, self.imageSize, self.nchannels)
        latent_inputs = tf.keras.Input(shape=(self.latentDim,), name='z_sampling')
        x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
        x = Reshape((shape[1], shape[2], shape[3]))(x)

        for i in range(self.nlayers):
            x = Conv2DTranspose(filters=filters,
                                 kernel_size=kernelSize,
                                 activation='relu',
                                 strides=2,
                                 padding='same')(x)
        filters //= 2

        outputs = Conv2DTranspose(filters=input_shape[2],
                                  kernel_size=kernel_size,
                                  activation='sigmoid',
                                  padding='same',
                                  name='decoder_output')(x)

        decoder = Model(latent_inputs, outputs)
        decoder.summary()