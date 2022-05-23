import tensorflow as tf
import time
from keras.callbacks import TerminateOnNaN, CSVLogger, ModelCheckpoint, EarlyStopping, TensorBoard
from os.path import join
import os

def decode_img(img):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize(img, [512, 512])


def process_path(file_path):
    # Load the raw data from the file as a string
    label = file_path
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

def readingData(inputDir,imageSize, batchSize, shuffle= True, validation = 0.0):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        inputDir,
        label_mode= None,
        seed = 123,
        image_size = (500, 500),
        validation_split= validation,
        subset='training',
        shuffle = shuffle,
        batch_size=batchSize,)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        inputDir,
        label_mode= None,
        seed = 123,
        image_size = (500, 500),
        validation_split= validation,
        subset='validation',
        shuffle = shuffle,
        batch_size=batchSize,)
    print('Reading in successful')
    resize_and_rescale = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Resizing(imageSize,imageSize),
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    ])
    augtrain_ds = train_ds.map(
        lambda x : (resize_and_rescale(x)), num_parallel_calls=tf.data.AUTOTUNE)
    augval_ds = val_ds.map(
        lambda x : (resize_and_rescale(x)), num_parallel_calls=tf.data.AUTOTUNE)
    return augtrain_ds, augval_ds



def readingDataWithFileNames(inputDir,imageSize):
    list_ds = tf.data.Dataset.list_files(inputDir + '/*/', shuffle = False)
    image_count = tf.data.experimental.cardinality(list_ds).numpy()

    image_ds = tf.keras.preprocessing.image_dataset_from_directory(
        inputDir,
        label_mode= None,
        image_size = (500, 500),
        shuffle = False)
    resize_and_rescale = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Resizing(imageSize,imageSize),
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    ])
    aug_ds = image_ds.map(
        lambda x : (resize_and_rescale(x)), num_parallel_calls=tf.data.AUTOTUNE)

    return aug_ds, list_ds, image_count

def createCallbacks(saveDir, earlystop, nlayers, learnRate, latentDim, nfilters, kernelSize):
    callbacks = []
    term_nan = TerminateOnNaN()
    callbacks.append(term_nan)
    csv_logger = CSVLogger(join(saveDir, 'training.log'), separator='\t')
    callbacks.append(csv_logger)
    checkpoint = ModelCheckpoint(join(saveDir, 'checkpoints/cvae_weights.hdf5'), verbose = 1, save_best_only=True, save_weights_only=True, monitor='loss')
    callbacks.append(checkpoint)
    tb_name = 'CVAE_nlayers' + str(nlayers) + '_lr' + str(learnRate) + '_latent' + str(latentDim) + '_filters' + str(nfilters) + '_kernel' + str(kernelSize) + '_{}'.format(int(time.time()))
    tb = TensorBoard(log_dir= join(saveDir, tb_name), histogram_freq=1, embeddings_freq = 2)
    callbacks.append(tb)
    if earlystop:
        earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=8)
        callbacks.append(earlystop)
    return callbacks

def createDirectories(foldername):
    os.mkdir(foldername)
    otherFolder = foldername + '/checkpoints'
    os.mkdir(otherFolder)
    return foldername + '/'

#%%
