import numpy as np
import tensorflow as tf
from models.utils import *


### Bayesian Neural Network Class
class BNN(object):
    def __init__(self, hyperParams):

        self.X = tf.placeholder("float", [None, hyperParams['input_d']])
        self.Y = tf.placeholder("float", [None, hyperParams['output_d']])

        self.prior = hyperParams['prior'] # for fixed prior
        # structure: ARD, ADD, ARD-ADD
        self.prior_struct = hyperParams['prior_structure'] 
        if self.prior_struct == "ard-add":
            self.taus = [1. for l in range(len(hyperParams['layer_sizes']) - 2)]
        # hyper prior distribution: inv Gamma, half-Cauchy
        self.prior_type = hyperParams['prior_type']
        # hyper prior params: alpha,beta / eta
        self.hyperprior = hyperParams['hyperprior']

        if hyperParams['is_ResNet']: self.fprop = fprop_bayes_resnet
        else: self.fprop = fprop_bayes_nn

        self.params = init_bnn(hyperParams['layer_sizes'])
        self.y_preds_linear = self.fprop(self.X, self.params, n_samples=hyperParams['n_samples'])
        self.elbo_obj, self.exp_ll, self.kld = self.get_ELBO()


    def get_ELBO(self, eps=0.):    

        # negative log likelihood terms
        ell_terms = []
        for y_hat in self.y_preds_linear:
            ell_terms.append( -compute_gauss_nll(self.Y, y_hat, noise_prec=20.) )
        all_ells = tf.concat(ell_terms, 1)

        # kld terms
        k = 0
        kld_terms = []
        for mu, sigma in zip(self.params['mu'], self.params['sigma']):
            
            if self.prior_struct == "none":
                opt_scale = self.prior['sigma']

            elif self.prior_struct == "add" and k > 0 and k < (len(self.params['mu'])-1):
                opt_scale = add_scale_update(self.prior_type, self.hyperprior, mu, tf.sqrt(sigma))

            elif self.prior_struct == "ard-add" and k > 0 and k < (len(self.params['mu'])-1):
                opt_scale = ard_add_scale_update(self.prior_type, self.hyperprior, self.taus[int(k-1)], mu, tf.sqrt(sigma))

            else:
                opt_scale = ard_scale_update(self.prior_type, self.hyperprior, mu, tf.sqrt(sigma))
                
            kld_terms.append(gauss2gauss_KLD(mu, tf.sqrt(sigma), self.prior['mu'], opt_scale))
            k += 1
            

        all_klds = tf.concat([tf.reshape(p, [1, -1]) for p in kld_terms], 1)
        
        final_ell_term = tf.reduce_mean(all_ells, reduction_indices=1, keep_dims=True)
        final_kld_term = tf.reduce_mean(all_klds)
        final_elbo = tf.reduce_mean(final_ell_term - final_kld_term)
        return final_elbo, tf.reduce_mean(final_ell_term), final_kld_term


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
