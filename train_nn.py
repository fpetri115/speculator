import os
os.environ["SPS_HOME"] = "/Users/fpetri/packages/fsps" 

#DELETE ABOVE TWO LINES ON HPC
#SWITCH OPTIMISER TO tf.keras.optimizers.Adam() ON HPC!!

import sys
import numpy as np
from speculator_mod.speculator import Photulator
import fsps
import matplotlib.pyplot as plt
import tensorflow as tf

select = int(sys.argv[1]) # select the filter to train
path = sys.argv[2]
file_id = sys.argv[3] #id for training data to use
ndata = int(sys.argv[4]) #input to get number of data to use in training
load_model = int(sys.argv[5]) # 1 -> load saved model, 0 -> don't 
patience = int(sys.argv[6]) # early stopping set up

lr = [float(i) for i in sys.argv[7].split()] #N
batch_size = [int(i) for i in sys.argv[8].split()] #N-1 (if add_final==1) N (if add_final==0)
gradient_accumulation_steps = [int(i) for i in sys.argv[9].split()] #N

add_final = int(sys.argv[10]) #1 = true: add run with batch_size=full dataset
max_epochs = int(sys.argv[11]) #max number of epochs to loop (set to 99999999 or some other large number if you want to stop via patience instead)

validation_split = float(sys.argv[12]) #fraction of training data used for validation

#check if GPU detected
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#get names of filters used
filter_list = fsps.find_filter('lsst')
print(filter_list)

#training data
spsparams = np.load(path+"/training_data/sps_parameters_"+file_id+".npy")[:ndata].astype(np.float32)
photometry = np.load(path+"/training_data/photometry_"+file_id+".npy")[:ndata].astype(np.float32)
print(spsparams.shape, photometry.shape)

# parameters shift and scale
parameters_shift = np.mean(spsparams, axis=0)
parameters_scale = np.std(spsparams, axis=0)
magnitudes_shift = np.mean(photometry, axis=0)
magnitudes_scale = np.std(photometry, axis=0)

#select approriate photometery given filter
filters = filter_list[select:select+1]
training_theta = tf.convert_to_tensor(spsparams)
training_mag = tf.convert_to_tensor(photometry[:,select:select+1])
print(filter)

# training set up
if(add_final == 1):
    batch_size.append(int((1-validation_split) * training_theta.shape[0]))
#BIGGER BATCH -> BETTER ESTIMATE OF GRADIENT BUT MORE MEMORY REQUIRED AND SLOWER 
#(REMEMBER: NETWORK ONLY UPDATES PER BATCH)
#   -A BIGGER BATCH MEANS YOU WILL LOOP THROUGH DATA QUICKER
#   -SO NEED MORE EPOCHS
# doing large batches last to check it doesnt crazy impact the validation loss? Do most of the work with small batches which is faster?

if(len(lr) != len(batch_size) or 
   len(lr) != len(gradient_accumulation_steps) or
     len(batch_size) != len(gradient_accumulation_steps)):
    raise Exception("miss matched size of lr/batch_size/gradient_acc") 

# architecture
n_layers = 4
n_units = 128

#extra params
verbose = True
restore_file = False
restore_filename = ''

#optimiser
optimiser = tf.keras.optimizers.legacy.Adam() #SWITCH TO tf.keras.optimizers.Adam() ON HPC!!

#running loss
running_loss = []
running_val_loss = [] #keeps track of validation loss across different batch sizes/learning rates

# architecture
n_hidden = [n_units]*n_layers

###############BEGIN TRAINIING

# train each band in turn
for f in range(len(filters)):

    if verbose is True:
        print('filter ' + filters[f] + '...')
    
    if(load_model == 1):
        restore_file = True
        restore_filename = path+'/trained_models/model_{}x{}'.format(n_layers, n_units) + filters[f]

    # construct the PHOTULATOR model
    photulator = Photulator(n_parameters=training_theta.shape[-1], 
                        filters=[filters[f]], 
                        parameters_shift=parameters_shift, 
                        parameters_scale=parameters_scale, 
                        magnitudes_shift=magnitudes_shift[f], 
                        magnitudes_scale=magnitudes_scale[f], 
                        n_hidden=n_hidden, 
                        restore=restore_file, 
                        restore_filename=restore_filename,
                        optimizer=optimiser)

    # train using cooling/heating schedule for lr/batch-size
    for i in range(len(lr)):
        
        if verbose is True:
            print('learning rate = ' + str(lr[i]) + ', batch size = ' + str(batch_size[i]), flush=True)

        # set learning rate
        photulator.optimizer.lr = lr[i]

        # split into validation and training sub-sets
        n_validation = int(training_theta.shape[0] * validation_split)
        n_training = training_theta.shape[0] - n_validation
        training_selection = tf.random.shuffle([True] * n_training + [False] * n_validation)

        # create iterable dataset (given batch size)
        train_mag = tf.expand_dims(training_mag[:,f],-1)
        training_data = tf.data.Dataset.from_tensor_slices((training_theta[training_selection], train_mag[training_selection])).shuffle(n_training).batch(batch_size[i])

        # set up training loss
        training_loss = [np.infty]
        validation_loss = [np.infty]
        best_loss = np.infty
        early_stopping_counter = 0
        epoch = 0

        # loop over epochs
        while early_stopping_counter < patience:

            # loop over batches for a single epoch 
            #for loop:(give one batch of param+phot -> update network once)
            for theta, mag in training_data:

                # training step: check whether to accumulate gradients or not (only worth doing this for very large batch sizes)
                if gradient_accumulation_steps[i] == 1:
                    loss = photulator.training_step(theta, mag)
                else:
                    #FURTHER BATCHING of batch into sub-batch as not to calculate the gradients all in one step, instead calculate them in smaller sub-batches
                    #but then still update the model in one batch (not sub-batches) as you have all the gradients, just 'acculumlated' to save memory
                    loss = photulator.training_step_with_accumulated_gradients(theta, mag, accumulation_steps=gradient_accumulation_steps[i])

                running_loss.append(loss)

            # compute total loss and validation loss
            validation_loss.append(photulator.compute_loss(training_theta[~training_selection], train_mag[~training_selection]).numpy())
            epoch+=1
            if verbose is True:
                    print('Running validation loss = ' + str(validation_loss[-1]), flush=True)

            # early stopping condition
            # if validation loss keeps going down, reset stopping counter
            if validation_loss[-1] < best_loss:
                best_loss = validation_loss[-1]
                early_stopping_counter = 0
            #else, if validation loss goes back up again, increment counter
            else:
                early_stopping_counter += 1

            #stop if has been training too long (fail safe for hpc)
            if(epoch > max_epochs):
                early_stopping_counter = patience
                print('Max Epochs Reached!', flush=True)
            #when counter reaches patience, save model(the larger patience, the more epochs in a row the validation loss needs to be same or increasing)
            if early_stopping_counter >= patience:
                photulator.update_emulator_parameters()
                photulator.save(path+'/trained_models/model_{}x{}'.format(n_layers, n_units) + filters[f])
                if verbose is True:
                    print('Validation loss = ' + str(best_loss), flush=True)
                running_val_loss.append(validation_loss)

if(load_model == 1):
    prev_loss = np.load(path+"/trained_models/loss_"+filters[f]+".npy")
    prev_val = np.load(path+"/trained_models/valloss_"+filters[f]+".npy")
    print(prev_val.shape, flush=True)
    new_loss = np.hstack((prev_loss, running_loss))
    new_val = np.hstack((prev_val, np.hstack(running_val_loss)))
    print(new_val.shape, flush=True)

    np.save(path+"/trained_models/loss_"+filters[f]+".npy", new_loss)
    np.save(path+"/trained_models/valloss_"+filters[f]+".npy", new_val)

else:
    np.save(path+"/trained_models/loss_"+filters[f]+".npy", running_loss)
    np.save(path+"/trained_models/valloss_"+filters[f]+".npy", np.hstack(running_val_loss))