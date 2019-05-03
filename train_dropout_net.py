import os
from os.path import join as pjoin
import _pickle as cp
from random import shuffle

import numpy as np
import tensorflow as tf

from models.dropoutnn import DropoutNN


# command line arguments
flags = tf.flags

### TRAINING ARGS
flags.DEFINE_integer("batchSize", 32, "batch size.")
flags.DEFINE_integer("nEpochs", 4500, "number of epochs to train.")
flags.DEFINE_float("adamLr", .001, "AdaM learning rate.")

### NETWORK STRUCTURE 
flags.DEFINE_integer("hidden_layers", 1, "number of hidden layers.")
flags.DEFINE_integer("hidden_size", 50, "number of hidden units in each layer.")
flags.DEFINE_integer("n_samples", 10, "number of samples to compute expected log likelihood.")
flags.DEFINE_float("prior_scale", 1., "prior's scale.")
flags.DEFINE_float("like_noise_prec", .75, "likelihood's test noise precision") #.75 for yacht
flags.DEFINE_string("is_resnet", "False", "contains residual layers")

### DATASET PARAMS
flags.DEFINE_string("dataset_path", "yacht_data/pkl/", "path to data set.")
flags.DEFINE_string("experimentDir", "yacht", "directory to save training artifacts.")
inArgs = flags.FLAGS


def get_file_name(expDir, dnnParams, trainParams):     
    # concat hyperparameters into file name
    output_file_base_name = '_'+''.join('{}_{}_'.format(key, val) for key, val in sorted(dnnParams.items()) if key not in ['prior', 'input_d', 'layer_sizes'])
    output_file_base_name += 'nLayers_'+str(len(dnnParams['layer_sizes']))+'_adamLR_'+str(trainParams['adamLr'])
                                                                               
    # check if results file already exists, if so, append a number                                                                                               
    results_file_name = pjoin(expDir, "/train_logs/dropnn_trainResultsIW"+output_file_base_name+".txt")
    file_exists_counter = 0
    while os.path.isfile(results_file_name):
        file_exists_counter += 1
        results_file_name = pjoin(expDir, "/train_logs/dropnn_trainResultsIW"+output_file_base_name+"_"+str(file_exists_counter)+".txt")
    if file_exists_counter > 0:
        output_file_base_name += "_"+str(file_exists_counter)

    return output_file_base_name


### Training function
def trainDNN(data, dnn_hyperParams, hyperParams, logFile=None, outfile_base_name="", y_scalers=None):

    n_splits, N_train, d = data['train']['x'].shape
    nTrainBatches = int(N_train/hyperParams['batchSize'])
    dnn_hyperParams['batchSize'] = hyperParams['batchSize']

    # init Dropout NN
    model = DropoutNN(dnn_hyperParams)

    # get training op
    optimizer = tf.train.AdamOptimizer(hyperParams['adamLr']).minimize(-model.exp_ll)

    test_log_likelihoods = []
    test_rmses = []
    with tf.Session(config=hyperParams['tf_config']) as s:

        for split_idx in range(n_splits):
            s.run(tf.initialize_all_variables())
            best_elbo = -10000000.
            best_epoch = 0

            ### TRAIN MODEL ###
            for epoch_idx in range(hyperParams['nEpochs']):
                #shuffle data after every epoch
                training_idxs = list(range(N_train))
                shuffle(training_idxs)
                data['train']['x'][split_idx, :, :] = data['train']['x'][split_idx, training_idxs, :]
                data['train']['y'][split_idx, :, :] = data['train']['y'][split_idx, training_idxs, :]

                # training
                best_ell = -np.inf
                exp_ll = 0.
                for batch_idx in range(nTrainBatches):
                    x = data['train']['x'][split_idx, batch_idx*hyperParams['batchSize']:(batch_idx+1)*hyperParams['batchSize'], :]
                    y = data['train']['y'][split_idx, batch_idx*hyperParams['batchSize']:(batch_idx+1)*hyperParams['batchSize'], :]
                    _, exp_ll_val, iws = s.run([optimizer, model.exp_ll, model.normalized_weights], {model.X: x, model.Y: y})
                    # if batch_idx == 0: print(str(os))
                    exp_ll += exp_ll_val

                # check for ELBO improvement
                star_printer = ""
                exp_ll /= nTrainBatches
                if exp_ll > best_ell: 
                    best_ell = exp_ll
                    best_epoch = epoch_idx
                    star_printer = "***"
               
                if (epoch_idx+1) % 500 == 0:
                    # log training progress
                    logging_str = "Epoch %d.  Expected LL: %.3f %s" %(epoch_idx+1, exp_ll, star_printer)
                    print(logging_str)
                    if logFile: 
                        logFile.write(logging_str + "\n")
                        logFile.flush()
                    #cp.dump(iws, open(inArgs.experimentDir+"/params/dropout_importance_weights.pkl", "wb"))

            ### SAVE WEIGHTS TO INSPECT SHRINKAGE BEHAVIOR
            if split_idx % 5 == 0:
                weight_matrices = {}
                weight_matrices = [s.run(m) for m in model.params['w']]
                cp.dump(weight_matrices, open(inArgs.experimentDir+"/params/dropout_weights"+outfile_base_name+"_splitIdx_"+str(split_idx)+".pkl", "wb"))

            ### TEST MODEL ###
            test_rmse, test_ll = s.run(model.get_test_metrics(500, y_mu=y_scalers[split_idx].mean_, y_scale=y_scalers[split_idx].scale_, likelihood_noise_prec=dnn_hyperParams['like_noise_prec']), {model.X: data['test']['x'][split_idx,:,:], model.Y: data['test']['y'][split_idx,:,:]})
            test_log_likelihoods.append( test_ll )
            test_rmses.append( test_rmse )
            logging_str = "\n\nRun #%d, Test RMSE: %.3f, Test Log Likelihood: %.3f \n\n" %(split_idx, test_rmses[-1], test_log_likelihoods[-1])
            print(logging_str)
            logging_file.write(logging_str+"\n")

    logging_str = "\n\n\n\n Avg Test RMSE: %.3f +- %.3f,  Avg Test Log Likelihood: %.3f +- %.3f" %(np.mean(test_rmses), np.std(test_rmses), np.mean(test_log_likelihoods), np.std(test_log_likelihoods))
    print(logging_str)
    logging_file.write(logging_str+"\n")


if __name__ == "__main__":

    # load UCI data set
    x_train, x_test, y_train, y_test, x_scalers, y_scalers = cp.load(open(inArgs.dataset_path+inArgs.experimentDir+".pkl", "rb"))
    
    uciData = {}
    uciData['train'] = {}
    uciData['test'] = {}
    uciData['train']['x'] = x_train
    uciData['test']['x'] = x_test
    uciData['train']['y'] = y_train
    uciData['test']['y'] = y_test

    # set architecture params
    dnn_hyperParams = {'input_d': uciData['train']['x'].shape[2], 'output_d': uciData['train']['y'].shape[2], 'n_samples':inArgs.n_samples, 'like_noise_prec':inArgs.like_noise_prec , 'prior':{'mu':0., 'sigma':inArgs.prior_scale}}
    dnn_hyperParams['layer_sizes'] = [dnn_hyperParams['input_d']] + [inArgs.hidden_size] * inArgs.hidden_layers + [dnn_hyperParams['output_d']]
    if inArgs.is_resnet == "True": dnn_hyperParams['is_ResNet'] = True
    else: dnn_hyperParams['is_ResNet'] = False
    
    # set training hyperparameters
    train_hyperParams = {'adamLr':inArgs.adamLr, 'nEpochs':inArgs.nEpochs, 'batchSize':inArgs.batchSize,  \
                         'tf_config': tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5), log_device_placement=False)}

    # setup files to write results 
    outfile_base_name = get_file_name(inArgs.experimentDir, dnn_hyperParams, train_hyperParams)
    logging_file = open(inArgs.experimentDir+"/train_logs/dropnn_trainResultsIW"+outfile_base_name+".txt", 'w')
    logging_file.write(str(dnn_hyperParams)+"\n\n")

    # train and evaluate model
    trainDNN(uciData, dnn_hyperParams, train_hyperParams, logging_file, outfile_base_name, y_scalers)

    logging_file.close()
