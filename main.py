'''
Part of PiTE
(c) 2023 by  Pengfei Zhang, Seojin Bang, Heewook Lee, and Arizona State University.
See LICENSE-CC-BY-NC-ND for licensing.
'''

#!/usr/bin/env python
# coding: utf-8

import sys
import os
import random
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import LambdaCallback,TensorBoard,ReduceLROnPlateau, EarlyStopping
from utils import print_performance

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

## Define arguments
parser = argparse.ArgumentParser()
# parser.add_argument('--setting', type=str,help='healthy_neg or wrong_neg')
parser.add_argument('--nns', type=str,help='transformer, bilstm, cnn, or baseline')
parser.add_argument('--split', type=str,help='tcr or epi')
parser.add_argument('--run', type=int)
parser.add_argument('--gpu', type=str)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

## Parameters setting    
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
seed = args.seed
np.random.seed(seed) 
BATCH_SIZE = 32

## Load model structures
if args.nns == 'transformer':
    from NNs import Transformer_based
    model = Transformer_based()
elif args.nns == 'bilstm':
    from NNs import BiLSTM_based
    model = BiLSTM_based()    
elif args.nns == 'cnn':
    from NNs import byteNet_based
    model = byteNet_based()    
elif args.nns == 'baseline':
    from NNs import baseline_net
    from utils import load_data_split
    model = baseline_net() 
else:
    print('Wrong neural network sturctures picked!')


## Training
print('Start training...')

if args.nns == 'baseline':
    '''
    Training steps for average pooling baseline model. The input size for the model is 1x1024 (post-averaging)
    '''
    # Read data
    # Change it to the path of your baseline data before run the code
    dat = pd.read_pickle("/mnt/disk07/user/pzhang84/data/tcr_repertoires_healthy_samples/combined_dataset_repTCRs/catELMo_combined.pkl")
    
    # Data split
    X1_train, X2_train, y_train, X1_test, X2_test, y_test, testData, trainData = load_data_split(dat, args.split, seed)
    
    checkpoint_filepath = 'models/' + args.nns + '_run_'+str(args.run)+'.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                    save_weights_only=True,
                                                                    monitor='val_loss',
                                                                    mode='min',
                                                                    save_best_only=True)
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 30)
    model.fit([X1_train,X2_train], y_train, verbose=1, validation_split=0.20, epochs=1, batch_size = 32, callbacks=[es, model_checkpoint_callback])
#     model.save('models/' + embedding_name + '.hdf5')
    yhat = model.predict([X1_test, X2_test])
    
    
    
else:
    '''
    Training steps for the rest models including our transformer-based, bilstm-based, and cnn-based.
    The input size for the model is 22x1024 (pre-averaging)
    '''
    ## Define DataLoader, otherwise it is difficult to load all training data at once
    if args.split == 'tcr':
        n_train, n_valid, n_test = 192063, 48016, 59937
    elif args.split == 'epi':
        n_train, n_valid, n_test = 186112, 46528, 67376

    class DataGenerator(keras.utils.Sequence):
        def __init__(self, nameDir, labels = None, batch_size=BATCH_SIZE, dim=22, n_channels=1024,
                     n_classes=2, shuffle=True):
            self.nameDir = nameDir
            if self.nameDir == 'train':
                self.list_IDs = np.arange(n_train)
            elif self.nameDir == 'valid':
                self.list_IDs = np.arange(n_valid)
            elif self.nameDir == 'test':
                self.list_IDs = np.arange(n_test)
            self.dim = dim
            self.batch_size = batch_size
            self.labels = labels

            self.n_channels = n_channels
            self.n_classes = n_classes
            self.shuffle = shuffle
            self.on_epoch_end()

        def __len__(self):
            return int(np.floor(len(self.list_IDs) / self.batch_size))

        def __getitem__(self, index):
            # Generate indexes of the batch
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
            # Find list of IDs
            list_IDs_temp = [self.list_IDs[k] for k in indexes]
            # Generate data
            [X_tcr, X_epi], y = self.__data_generation(list_IDs_temp)
            return [X_tcr, X_epi], y

        def on_epoch_end(self):
            self.indexes = np.arange(len(self.list_IDs))
            if self.shuffle == True:
                np.random.shuffle(self.indexes)

        def __data_generation(self, list_IDs_temp):
            # Initialization
            X_tcr = np.empty((self.batch_size, self.dim, self.n_channels))
            X_epi = np.empty((self.batch_size, self.dim, self.n_channels))
            y = np.empty((self.batch_size), dtype=int)

            # Generate data
            for i, ID in enumerate(list_IDs_temp):
                # Load npz files
                # Change it to the path of your pre-averaged data before run the code
                npzfile = np.load('../data_healthy_neg/seed42_'+ args.split + '_split/' + self.nameDir + '/' + str(ID) + '.npz')
                X_tcr[i,] = npzfile['x_tcr']
                X_epi[i,] = npzfile['x_epi']
                # Store class
                y[i] = npzfile['y']
            return [X_tcr, X_epi], y
    
    ## Load training and validation npz data
    traingen = DataGenerator('train')
    validgen = DataGenerator('valid')

    ## Training the model
    model_path = './models'
    logs_path = './logs'

    exp_name = args.nns+args.split+'_run_'+str(args.run)

    tbCallBack = TensorBoard(log_dir=os.path.join(logs_path, exp_name),
                             histogram_freq=0,
                             write_graph=True, 
                             write_images=True,
                            )
    tbCallBack.set_model(model)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                   patience=30, 
                                                   verbose=1,
                                                   mode='min',
                                                  )
    check_point = keras.callbacks.ModelCheckpoint(os.path.join(model_path, exp_name + ".h5"),
                                                  monitor='val_loss', 
                                                  verbose=1, 
                                                  save_best_only=True, 
                                                  mode='min',
                                                 )
    lrate_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10,
                                        min_delta=0.0001, min_lr=1e-6, verbose=1)

    callbacks = [check_point, early_stopping, tbCallBack, lrate_scheduler]

    model.fit_generator(traingen, validation_data = validgen, callbacks=callbacks, 
              use_multiprocessing=True, max_queue_size=128, workers = 12, verbose = 1, epochs = 1)

    ## Evaluation the trained model on the testing data
    print('Evaluating...')
    print('Loading testing data...')
    testData = pd.read_pickle('../data_healthy_neg/seed42_'+args.split+'_split/testData_labels.pkl')
    testgen = DataGenerator('test', shuffle=False, batch_size=1)
    yhat = model.predict_generator(testgen)

    
## Report performance including AUC, Recall, Precision, and F1 scores.
y = np.array(testData.binding.to_list())
print_performance(y, yhat)

## Save the predicted results
# testData['yhat'] = yhat
# testData.to_pickle(args.split+'_'+args.nns + '.pkl')
