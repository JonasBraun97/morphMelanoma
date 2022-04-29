import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, InputLayer, Conv2DTranspose, BatchNormalization, Activation
from tensorflow.keras.callbacks import TerminateOnNaN, CSVLogger, ModelCheckpoint, EarlyStopping

class CVAE(tf.keras.Model):
    """n-dim variational Autoencoder for latent phenotype capture"""
    def __init__(self, args):
        self.inputDir = args.inputDir
        self.outputDir = args.outputDir
        self.saveDir = args.saveDir

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


            self.decoder = tf.keras.Sequential(
                [
                    InputLayer(input_shape=(latent_dim,)),
                    Dense(units=shape[1]*shape[2]*shape[3], activation='relu'),
                    Reshape(target_shape=(shape[1], shape[2], shape[3]))
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

        #save model architectures
        self.model_dir = os.path.join(self.save_dir, 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        print('saving model architectures to', self.model_dir)
        with open(os.path.join(self.model_dir, 'arch_vae.json'), 'w') as file:
            file.write(self.vae.to_json())
        with open(os.path.join(self.model_dir, 'arch_encoder.json'), 'w') as file:
            file.write(self.encoder.to_json())
        with open(os.path.join(self.model_dir, 'arch_decoder.json'), 'w') as file:
            file.write(self.decoder.to_json())

    def train(self):
        """ train VAE model
        """
        #Note: Datagenerator including on the fly rotation, orientation, and size transformations not included here. Implement according to needs and use to load images.

        print('Loading Input1 Images')
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.inputDir,
            label_mode= None,
            validation_split = 0.2,
            subset = "training",
            seed = 123,
            image_size = (self.imageSize, self.imageSize),
            batch_size=self.batchSize,)

        callbacks = []
        #terminates when the loss is NaN
        term_nan = TerminateOnNaN()
        callbacks.append(term_nan)

        #save callbacks in a csv file
        csv_logger = CSVLogger(os.path.join(self.save_dir, 'training.log'),
                           separator='\t')
        callbacks.append(csv_logger)

        checkpointer = ModelCheckpoint(
            os.path.join(self.save_dir, 'checkpoints/vae_weights.hdf5'),
            verbose=1,
            save_best_only=True,
            save_weights_only=True)
        callbacks.append(checkpointer)

        if self.earlystop:
            earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=8)
            callbacks.append(earlystop)


        #if self.use_vaecb:
            #vaecb = VAEcallback(self)
            #callbacks.append(vaecb)

        self.history = self.vae.fit(
            x={"encoder_input1": self.input1_data, "encoder_input2": self.input2_data},
            y=out1_data,
            epochs = self.epochs,
            callbacks = callbacks,
            batch_size=self.batch_size)


        # imageList=sorted(glob.glob(os.path.join(self.data_dir,'train', '*')))

        #data = []
        #for imagePath in imageList:
            #image = imread(imagePath)
            #image=resize(image,(self.image_size, self.image_size, self.nchannel))
            #image=image*(255/np.max(image))
            #data.append(image)
        #self.input1_data = np.array(data, dtype="float") / 255.0


        #print('Loading Output Images')
        #imageList=sorted(glob.glob(os.path.join(out1_dir,'train', '*')))
        #data = []
        #for imagePath in imageList:
            #image = imread(imagePath)
            #image=resize(image,(self.image_size, self.image_size, self.nchannel))
            #image=image*(255/np.max(image))
            #data.append(image)
        #out1_data = np.array(data, dtype="float") / 255.0

        print('saving model weights to', self.model_dir)
        self.vae.save_weights(os.path.join(self.model_dir, 'weights_vae.hdf5'))
        self.encoder.save_weights(os.path.join(self.model_dir, 'weights_encoder.hdf5'))
        self.decoder.save_weights(os.path.join(self.model_dir, 'weights_decoder.hdf5'))
        self.encode()

        print('done!')





