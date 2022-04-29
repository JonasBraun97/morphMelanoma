import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, InputLayer, Conv2DTranspose, BatchNormalization, Activation

class CVAE(tf.keras.Model):
    """n-dim variational Autoencoder for latent phenotype capture"""
    def __init__(self, args):
        self.inputDir = args.inputDir
        self.outputDir = args.outputDir

        #callback inputs
        self.cvaecbUsage = args.cvaecbUsage
        self.earlystop = args.earlystop

        self.epochs = args.epochs
        self.learnRate = args.learnRate
        self.latenDim = args.latentDim
        self.nlayers = args.nlayers
        self.interDim = args.interDim
        self.kernelSize = args.kernelSize
        self.batchSize = args.batchSize
        self.epsilonStd = args.epsilonStd

        self.trainpath =os.path.join(self.inputDir, 'train'))
        self.dataSizeTrain = len(self.trainpath)
        self.files = [f for f in listdir(self.trainpath) if isfile(join(trainpath, f))]

        self.imageSize = args.imageSize
        self.nchannels = args.nchannels


        def build_model(self):
            inputShape = (self.imageSize, self.imageSize, self.nchannels)
            filters = self.nfilters
            kernelSize = self.kernelSize
            interDim = self.interDim
            latentDim = self.latentDim

            self.encoder = tf.keras.Sequential(
                [
                    InputLayer(input_shape=inputShape)

                ]
            )

            #add convolutional layers to Encoder
            for i in range(self.nlayers):
                self.encoder.add(
                    Conv2D(
                        filters = filters,
                        kernel_size= kernelSize,
                        activation='relu'
                        strides = 2,
                        padding = 'same'
                    )
                )
                filters *= 2

            self.encoder.add(
                [
                    Flatten(),
                    Dense(interDim + interDim)
                ]
            )

            # generate latent vector Q(z|X)
            self.latent = tf.keras.Sequential(
                [
                    z2_mean = Dense(self.latent_dim, name='z_mean')(x2)
            z2_log_var = Dense(self.latent_dim, name='z_log_var')(x2)

            # use reparameterization trick to push the sampling out as input
            z2 = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z2')([z2_mean, z2_log_var])

            z12=Multiply()([z1,z2])
                ]
            )


            self.decoder = tf.keras.Sequential(
                [
                    InputLayer(input_shape=(latent_dim,)),
                    Dense(units=125*125*32, activation='relu'),
                    Reshape(target_shape=(125, 125, 32))
                ]
            )

            for i in range(self.nlayers):
                self.decoder.add(
                    Conv2DTranspose(
                        filters = filters,
                        kernel_size= kernelSize,
                        activation='relu',
                        strides = 2,
                        padding = 'same'
                    )
                )
                filters //= 2

            self.decoder.add(
                Conv2DTranspose(
                    filters=input_shape[2],
                    kernel_size=kernel_size,
                    activation='sigmoid',
                    padding='same',
                )

            )

            self.encoder.summary()
            plot_model(self.encoder, to_file=os.path.join(self.save_dir, 'encoder_model.png'), show_shapes=True)

            self.decoder.summary()
            plot_model(self.decoder, to_file=os.path.join(self.save_dir, 'decoder_model.png'), show_shapes=True)




        optimizer = tf.keras.optimizers.Adam(self.learnRate)

        def sample(self, eps=None):
            if eps is None:
                eps = tf.random.normal(shape=(100, self.latent_dim))
            return self.decode(eps, apply_sigmoid=True)

        def encode(self, x):
            mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
            return mean, logvar

        def

        def reparametrize(self, sample_args):
            #put this maybe into self
            zmean, zvar = sample_args

            eps = tf.random_normal(shape=(K.shape(zmean)[0],
                                          self.latentDim),
                                   mean=0,
                                   stddev=self.epsilonStd)
            return eps * tf.exp(0.5 * zvar) +zmean

        def decode(self, z, apply_sigmoid=False):
            logits = self.decoder(z)
            if apply_sigmoid:
                probs = tf.sigmoid(logits)
                return probs
            return logits

        def get_latent(self, x):
            return self.encoder(x)

        def compute_loss(model, x, self):
            z = self.encode(x)
            mean, logvar = model.encode(x)
            z = model.reparameterize(mean, logvar)
            x_logit = model.decode(z)
            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
            logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
            logpz = log_normal_pdf(z, 0., 0.)
            logqz_x = log_normal_pdf(z, mean, logvar)
            return -tf.reduce_mean(logpx_z + logpz - logqz_x)





