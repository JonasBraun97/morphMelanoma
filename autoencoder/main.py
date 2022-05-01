""" 
Image Variational Autoencoding
"""

import sys
import os
import argparse
from cvae import CVAE
import tensorflow as tf
from tensorflow.keras.callbacks import TerminateOnNaN, CSVLogger, ModelCheckpoint, EarlyStopping

parser = argparse.ArgumentParser(description='')
parser.add_argument('--inputDir',       type=str,   default='InputImages1/',     help='input data directory (in train subfolder)')
parser.add_argument('--outputDir',       type=str,   default='OutputImages/',     help='output 2 data directory (in train subfolder)')
parser.add_argument('--saveDir',       type=str,   default='Outputs/',     help='save directory')

parser.add_argument('--earlystop',		type=int,	default=1,			help='use early stopping? 1=yes, 0=no')

parser.add_argument('--epochs',         type=int,   default=10,          help='training epochs')
parser.add_argument('--learnRate',     type=float, default=0.0001,      help='learning rate')
parser.add_argument('--latentDim',     type=int,   default=128,          help='latent dimension')
parser.add_argument('--interDim',      type=int,   default=128,        help='intermediate dimension')
parser.add_argument('--nlayers',        type=int,   default=3,          help='number of layers in models')
parser.add_argument('--nfilters',       type=int,   default=16,         help='num convolution filters')
parser.add_argument('--kernelSize',    type=int,   default=3,          help='number of convolutions')
parser.add_argument('--batchSize',     type=int,   default=50,         help='batch size')
parser.add_argument('--epsilonStd',    type=float, default=1.0,        help='epsilon width')


parser.add_argument('--imageSize',     type=int,   default=500,         help='image size')
parser.add_argument('--nchannels',       type=int,   default=3,          help='image channels')

parser.add_argument('--phase',          type=str,   default='train',    help='train or load')

args = parser.parse_args()


def main():

    os.makedirs(args.saveDir, exist_ok=True)

    if args.phase == 'train':
        model = CVAE(args)
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            args.inputDir,
            label_mode= None,
            validation_split = 0.2,
            subset = "training",
            seed = 123,
            image_size = (args.imageSize, args.imageSize),
            batch_size=args.batchSize,)

        callbacks = []
        term_nan = TerminateOnNaN()
        callbacks.append(term_nan)
        csv_logger = CSVLogger(os.path.join(args.saveDir, 'training.log'),
                               separator='\t')
        callbacks.append(csv_logger)
        checkpointer = ModelCheckpoint(
            os.path.join(args.saveDir, 'checkpoints/vae_weights.hdf5'),
            verbose=1,
            save_best_only=True,
            save_weights_only=True)
        callbacks.append(checkpointer)

        if args.earlystop:
            earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=8)
            callbacks.append(earlystop)

        model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(args.learnRate))

        model.fit(
            x=train_ds,
            y=train_ds,
            epochs = args.epochs,
            callbacks = callbacks,
            batch_size=args.batchSize)

        model.save_weights()

    if args.phase == 'load':
        if args.checkpoint == 'NA':
            sys.exit('No checkpoint file provided')
        model = MEVAE(args)
        model.vae.load_weights(args.checkpoint)
        model.train()
        
    
if __name__ == '__main__':
    main()
