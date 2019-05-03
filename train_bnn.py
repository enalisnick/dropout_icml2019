import os
from os.path import join as pjoin
import _pickle as cp
from random import shuffle

import numpy as np
import tensorflow as tf

from models.bnn import BNN


# command line arguments
flags = tf.flags

### TRAINING ARGS
flags.DEFINE_integer("batchSize", 32, "batch size.")
flags.DEFINE_integer("nEpochs", 4500, "number of epochs to train.")
flags.DEFINE_float("adamLr", .001, "AdaM learning rate.")

### NETWORK STRUCTURE 
flags.DEFINE_integer("hidden_layers", 2, "number of hidden layers.")
flags.DEFINE_integer("hidden_size", 50, "number of hidden units in each layer.")
flags.DEFINE_integer("n_samples", 10, "number of samples to compute expected log likelihood.")
flags.DEFINE_float("prior_scale", 1., "prior's scale.")
flags.DEFINE_float("like_noise_prec", 0.75, "likelihood's test noise precision")
flags.DEFINE_string("is_resnet", "True", "contains residual layers")

### HYPER-PRIOR PARAMS
flags.DEFINE_string("prior_struct", "ard-add", "scale tying structure")
flags.DEFINE_string("hyperprior_dist", "inv_gamma", "distribution on scale")
flags.DEFINE_float("hyperprior_param1", 3., "hyper-prior param #1")
flags.DEFINE_float("hyperprior_param2", 3., "hyper-prior param #2")

### DATASET PARAMS
flags.DEFINE_string("dataset_path", "yacht_data/pkl/", "path to data set.")
flags.DEFINE_string("experimentDir", "yacht", "directory to save training artifacts.")
inArgs = flags.FLAGS


def get_file_name(expDir, bnnParams, trainParams):     
    # concat hyperparameters into file name
    output_file_base_name = '_'+''.join('{}_{}_'.format(key, val) for key, val in sorted(bnnParams.items()) if key not in ['prior', 'input_d', 'layer_sizes', 'hyperprior'])
    output_file_base_name += 'nLayers_'+str(len(bnnParams['layer_sizes']))+'_adamLR_'+str(trainParams['adamLr'])
                                                                               
    # check if results file already exists, if so, append a number                                                                                               
    results_file_name = pjoin(expDir, "/train_logs/bnn_trainResults"+output_file_base_name+".txt")
    file_exists_counter = 0
    while os.path.isfile(results_file_name):
        file_exists_counter += 1
        results_file_name = pjoin(expDir, "/train_logs/bnn_trainResults"+output_file_base_name+"_"+str(file_exists_counter)+".txt")
    if file_exists_counter > 0:
        output_file_base_name += "_"+str(file_exists_counter)

    return output_file_base_name


### Training function
def trainBNN(data, bnn_hyperParams, hyperParams, logFile=None, outfile_base_name="", y_scalers=None):

    n_splits, N_train, d = data['train']['x'].shape
    nTrainBatches = int(N_train/hyperParams['batchSize'])
    bnn_hyperParams['batchSize'] = hyperParams['batchSize']

    # init Bayes NN
    model = BNN(bnn_hyperParams)

    # get training op
    optimizer = tf.train.AdamOptimizer(hyperParams['adamLr']).minimize(-model.elbo_obj)

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
                train_elbo = 0.
                exp_ll = 0.
                kld = 0.
                for batch_idx in range(nTrainBatches):
                    x = data['train']['x'][split_idx, batch_idx*hyperParams['batchSize']:(batch_idx+1)*hyperParams['batchSize'], :]
                    y = data['train']['y'][split_idx, batch_idx*hyperParams['batchSize']:(batch_idx+1)*hyperParams['batchSize'], :]
                    _, elbo_val, exp_ll_val, kld_val = s.run([optimizer, model.elbo_obj, model.exp_ll, model.kld], {model.X: x, model.Y: y})
                    # if batch_idx == 0: print(str(os))
                    train_elbo += elbo_val
                    exp_ll += exp_ll_val
                    kld += kld_val

                # check for ELBO improvement
                star_printer = ""
                train_elbo /= nTrainBatches
                exp_ll /= nTrainBatches
                kld /= nTrainBatches
                if train_elbo > best_elbo: 
                    best_elbo = train_elbo
                    best_epoch = epoch_idx
                    star_printer = "***"
               
                if (epoch_idx+1) % 500 == 0:
                    # log training progress
                    logging_str = "Epoch %d.  Expected LL: %.3f,  KLD: %.3f,  Train ELBO: %.3f %s" %(epoch_idx+1, exp_ll, kld, train_elbo, star_printer)
                    print(logging_str)
                    if logFile: 
                        logFile.write(logging_str + "\n")
                        logFile.flush()

            ### SAVE WEIGHTS TO INSPECT SHRINKAGE BEHAVIOR
            if split_idx % 5 == 0:
                weight_matrices = {}
                weight_matrices['mus'] = [s.run(m) for m in model.params['mu']]
                weight_matrices['sigmas'] = [s.run(sig) for sig in model.params['sigma']]
                cp.dump(weight_matrices, open(inArgs.experimentDir+"/params/weights_post_"+outfile_base_name+"_splitIdx_"+str(split_idx)+".pkl", "wb"))

            ### TEST MODEL ###
            test_rmse, test_ll = s.run(model.get_test_metrics(500, y_mu=y_scalers[split_idx].mean_, y_scale=y_scalers[split_idx].scale_, likelihood_noise_prec=bnn_hyperParams['like_noise_prec']), {model.X: data['test']['x'][split_idx,:,:], model.Y: data['test']['y'][split_idx,:,:]})
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
    bnn_hyperParams = {'input_d': uciData['train']['x'].shape[2], 'output_d': uciData['train']['y'].shape[2], 'n_samples':inArgs.n_samples, 'like_noise_prec':inArgs.like_noise_prec , 'prior':{'mu':0., 'sigma':inArgs.prior_scale}}
    bnn_hyperParams['layer_sizes'] = [bnn_hyperParams['input_d']] + [inArgs.hidden_size] * inArgs.hidden_layers + [bnn_hyperParams['output_d']]
    if inArgs.is_resnet == "True": bnn_hyperParams['is_ResNet'] = True
    else: bnn_hyperParams['is_ResNet'] = False

    # set hyper prior
    bnn_hyperParams['prior_structure'] = inArgs.prior_struct
    bnn_hyperParams['prior_type'] = inArgs.hyperprior_dist
    bnn_hyperParams['hyperprior'] = {"param1":inArgs.hyperprior_param1, "param2":inArgs.hyperprior_param2}
    
    # set training hyperparameters
    train_hyperParams = {'adamLr':inArgs.adamLr, 'nEpochs':inArgs.nEpochs, 'batchSize':inArgs.batchSize,  \
                         'tf_config': tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5), log_device_placement=False)}

    # setup files to write results 
    outfile_base_name = get_file_name(inArgs.experimentDir, bnn_hyperParams, train_hyperParams)
    logging_file = open(inArgs.experimentDir+"/train_logs/bnn_trainResults"+outfile_base_name+".txt", 'w')
    logging_file.write(str(bnn_hyperParams)+"\n\n")

    # train and evaluate model
    trainBNN(uciData, bnn_hyperParams, train_hyperParams, logging_file, outfile_base_name, y_scalers)

    logging_file.close()
