import numpy as np
import tensorflow as tf
from models.utils import *


### Dropout Neural Network Class
class DropoutNN(object):
    def __init__(self, hyperParams):

        self.X = tf.placeholder("float", [None, hyperParams['input_d']])
        self.Y = tf.placeholder("float", [None, hyperParams['output_d']])

        if hyperParams['is_ResNet']: self.fprop = fprop_dropout_resnet
        else: self.fprop = fprop_dropout_nn

        self.params = init_nn(hyperParams['layer_sizes'])
        self.y_preds_linear = self.fprop(self.X, self.params, n_samples=hyperParams['n_samples'])
        self.exp_ll, self.normalized_weights = self.get_ELBO()


    def get_ELBO(self, eps=0.):    

        # negative log likelihood terms
        ell_terms = []
        for y_hat in self.y_preds_linear:
            ell_terms.append( -compute_gauss_nll(self.Y, y_hat, noise_prec=10.) )
        all_ells = tf.concat(ell_terms, 1)
        
        # sort the samples
        all_ells, _ = tf.nn.top_k(all_ells, k=10, sorted=True)

        # create rank-based weights
        gamma_vals = 10./(tf.expand_dims(tf.range(10, dtype=tf.float32), 0) + 1.)
        normalized_weights = tf.stop_gradient(gamma_vals / tf.reduce_sum(gamma_vals, axis=1, keep_dims=True))
        
        # apply those weights as a constant
        final_ell_term = tf.reduce_sum(normalized_weights * all_ells, reduction_indices=1, keep_dims=True)
        
        return tf.reduce_mean( final_ell_term ), normalized_weights


    def get_test_metrics(self, nSamples, y_mu, y_scale, likelihood_noise_prec=.75):
        
        lls = []
        rmses = []
        y_hats = self.fprop(self.X, self.params, n_samples=nSamples) 
        for y_hat in y_hats:
            y_hat_scaled_trans = y_hat * y_scale + y_mu
            lls.append( -compute_gauss_nll(self.Y, y_hat_scaled_trans, likelihood_noise_prec) )
            rmses.append( tf.sqrt(tf.reduce_mean(tf.square(self.Y - y_hat_scaled_trans), reduction_indices=1, keep_dims=True)) )
        
        all_lls = tf.concat(lls, 1)
        all_rmses = tf.concat(rmses, 1)
        maxs = tf.reduce_max(all_lls, reduction_indices=1, keep_dims=True)

        return tf.reduce_mean(tf.reduce_mean(all_rmses, reduction_indices=1)), tf.reduce_mean(tf.log(tf.reduce_mean(tf.exp(all_lls - maxs), reduction_indices=1, keep_dims=True)) + maxs)
