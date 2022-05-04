import tensorflow as tf
from cvae import CVAE
import time
import argparse
from keras.callbacks import TerminateOnNaN, CSVLogger, ModelCheckpoint, EarlyStopping, TensorBoard
from os.path import join
import os
from tensorboard import program

parser = argparse.ArgumentParser(description='')
parser.add_argument('--inputDir',       type=str,   default='/Users/jones/Library/CloudStorage/OneDrive-NorthwesternUniversity/ownProject/data/mergedDAPIYFPNormalized_Subregion_13_r2_c3/singleCellImages',     help='input data directory (in train subfolder)')
#parser.add_argument('--outputDir',       type=str,   default='OutputImages/',     help='output 2 data directory (in train subfolder)')
#parser.add_argument('--saveDir',       type=str,   default='Outputs/',     help='save directory')

parser.add_argument('--earlystop',		type=int,	default=1,			help='use early stopping? 1=yes, 0=no')

parser.add_argument('--epochs',         type=int,   default=2,          help='training epochs')
parser.add_argument('--learnRate',     type=float, default=0.0001,      help='learning rate')
#parser.add_argument('--latentDim',     type=int,   default=128,          help='latent dimension')
parser.add_argument('--interDim',      type=int,   default=128,        help='intermediate dimension')
#parser.add_argument('--nlayers',        type=int,   default=2,          help='number of layers in models')
parser.add_argument('--nfilters',       type=int,   default=16,         help='num convolution filters')
#parser.add_argument('--kernelSize',    type=int,   default=3,          help='number of convolutions')
parser.add_argument('--batchSize',     type=int,   default=50,         help='batch size')
parser.add_argument('--epsilonStd',    type=float, default=1.0,        help='epsilon width')


parser.add_argument('--imageSize',     type=int,   default=512,         help='image size')
parser.add_argument('--nchannel',       type=int,   default=3,          help='image channels')

parser.add_argument('--phase',          type=str,   default='train',    help='train or load')

args = parser.parse_args()

layersList = [3,4]
latentList = [128,64]
#test latentDim < interDim
interDimList = [128, 64]
kernelList = [3]
learnRateList = [0.0001, 0.001]
filtersList = [16, 32]

def main():
    runNumber = 0
    for i in range(len(layersList)):
        for j in range(len(latentList)):
            for k in range(len(kernelList)):
                args.nlayers = layersList[i]
                args.latentDim = latentList[j]
                args.kernelSize = kernelList[k]
                foldername = 'Outputs' + str(runNumber)
                os.mkdir(foldername)
                otherFolder = foldername + '/checkpoints'
                os.mkdir(otherFolder)
                args.saveDir = foldername + '/'
                print('Created new Directory to store Output')
                #macbookPathImage = '/Users/jones/Library/CloudStorage/OneDrive-NorthwesternUniversity/ownProject/data/mergedDAPIYFPNormalized_Subregion_13_r2_c3/singleCellImages'
                print('Reading in images')
                train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                    args.inputDir,
                    label_mode= None,
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
                checkpoint = ModelCheckpoint(join(args.saveDir, 'checkpoints/cvae_weights.hdf5'), verbose = 1, save_best_only=True, save_weights_only=True, monitor='loss')
                callbacks.append(checkpoint)
                tb_name = 'CVAE_nlayers' + str(args.nlayers) + '_lr' + str(args.learnRate) + '_latent' + str(args.latentDim) + '_filters' + str(args.nfilters) + '_kernel' + str(args.kernelSize) + '_{}'.format(int(time.time()))
                tb = TensorBoard(log_dir= join(args.saveDir, tb_name), histogram_freq=1, embeddings_freq = 2)
                callbacks.append(tb)

                if args.earlystop:
                    earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=8)
                    callbacks.append(earlystop)

                optimizer = tf.keras.optimizers.Adam(args.learnRate)
                # set the dimensionality of the latent space to a plane for visualization later
                num_examples_to_generate = 16
                random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, args.latentDim])
                print('Init model')
                model = CVAE(args)
                print('Compile model')
                model.compile(optimizer = optimizer)
                print('Model compiled. model.fit starts')
                model.fit(aug_ds, epochs = args.epochs, batch_size=args.batchSize, callbacks=callbacks)


                modelDir = os.path.join(args.saveDir, 'models')
                os.makedirs(modelDir, exist_ok=True)
                print('Save model weights to', modelDir)
                model.save_weights(join(modelDir, 'weights_CVAE.hdf5'))

                print('Model saved. Start encoding')
                model.encode(aug_ds)
                print('DONE!!')
                runNumber += 1


if __name__ == '__main__':
    main()