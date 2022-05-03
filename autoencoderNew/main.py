import tensorflow as tf
from cvae import CVAE
import time
import argparse
from keras.callbacks import TerminateOnNaN, CSVLogger, ModelCheckpoint, EarlyStopping

parser = argparse.ArgumentParser(description='')
parser.add_argument('--inputDir',       type=str,   default='/Users/jones/Library/CloudStorage/OneDrive-NorthwesternUniversity/ownProject/data/mergedDAPIYFPNormalized_Subregion_13_r2_c3/singleCellImages',     help='input data directory (in train subfolder)')
#parser.add_argument('--outputDir',       type=str,   default='OutputImages/',     help='output 2 data directory (in train subfolder)')
parser.add_argument('--saveDir',       type=str,   default='Outputs/',     help='save directory')

parser.add_argument('--earlystop',		type=int,	default=1,			help='use early stopping? 1=yes, 0=no')

parser.add_argument('--epochs',         type=int,   default=10,          help='training epochs')
parser.add_argument('--learnRate',     type=float, default=0.0001,      help='learning rate')
parser.add_argument('--latentDim',     type=int,   default=128,          help='latent dimension')
parser.add_argument('--interDim',      type=int,   default=128,        help='intermediate dimension')
parser.add_argument('--nlayers',        type=int,   default=2,          help='number of layers in models')
parser.add_argument('--nfilters',       type=int,   default=16,         help='num convolution filters')
parser.add_argument('--kernelSize',    type=int,   default=3,          help='number of convolutions')
parser.add_argument('--batchSize',     type=int,   default=50,         help='batch size')
parser.add_argument('--epsilonStd',    type=float, default=1.0,        help='epsilon width')


parser.add_argument('--imageSize',     type=int,   default=512,         help='image size')
parser.add_argument('--nchannels',       type=int,   default=3,          help='image channels')

parser.add_argument('--phase',          type=str,   default='train',    help='train or load')

args = parser.parse_args()


def main():
    #macbookPathImage = '/Users/jones/Library/CloudStorage/OneDrive-NorthwesternUniversity/ownProject/data/mergedDAPIYFPNormalized_Subregion_13_r2_c3/singleCellImages'
    print('Reading in images')
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        args.inputDir,
        label_mode= None,
        validation_split = 0.2,
        subset = "training",
        seed = 123,
        image_size = (500, 500),
        batch_size=args.batchSize,)
    print('Reading in successful')
    resize_and_rescale = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Resizing(args.imageSize,args.imageSize),
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    ])
    aug_ds = train_ds.map(
        lambda x : (resize_and_rescale(x)))
    print('images transformed')

    print('Create callbacks')
    callbacks = []
    term_nan = TerminateOnNaN()
    callbacks.append(term_nan)
    csv_logger = CSVLogger(join(args.saveDir, 'training.log'), separator='\t')
    callbacks.append(csv_logger)
    checkpoint = ModelCheckpoint(join(args.saveDir, 'checkpoints/cvae_weights.hdf5'), verbose = 1, save_best_only=True, save_weights_only=True)
    callbacks.append(checkpoint)

    if args.earlystop:
        earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=8)
        callbacks.append(earlystop)

    optimizer = tf.keras.optimizers.Adam(args.learnRate)
    # set the dimensionality of the latent space to a plane for visualization later
    # num_examples_to_generate = 16
    random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, args.latentDim])
    print('Init model')
    model = CVAE()
    print('Compile model')
    model.compile(optimizer = optimizer)
    print('Model compiled. model.fit starts')
    model.fit(aug_ds, epochs = args.epochs, batch_size=args.batchSize, callbacks=callbacks)

    print('Save model weights to', args.modelDir)
    model.save_weights(join(args.modelDir, 'weights_CVAE.hdf5'))
    print('DONE!!')


if __name__ == '__main__':
    main()