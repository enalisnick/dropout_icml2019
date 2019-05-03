import numpy as np
import tensorflow as tf


def init_nn(layer_sizes, std=.01):
    params = {'w':[], 'b':[]}
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        params['w'].append(tf.Variable(tf.random_uniform([n_in, n_out], minval=-.05, maxval=.05)))
        params['b'].append(tf.Variable(tf.zeros([n_out,]) + .1))
    return params

                                                                                                                  
def init_bnn(layer_sizes, std=.01):
    params = {'mu':[], 'sigma':[], 'b':[]}
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        params['mu'].append(tf.Variable(tf.random_uniform([n_in, n_out], minval=-.05, maxval=.05)))
        params['sigma'].append(tf.nn.softplus(tf.Variable(tf.random_uniform([n_in, n_out], minval=-10.0, maxval=-6.5))))
        params['b'].append(tf.Variable(tf.zeros([n_out,]) + .1))
    return params


def fprop_bayes_resnet(X, params, n_samples=10):
# batch_size = tf.shape(X)[0]                                                                                                                                                                                                                           
    hidden_states = []
    # take multiple samples at first layer                                                                                                                                                                                                          
    for s_idx in range(n_samples):
        S = tf.nn.dropout(tf.ones_like(X), keep_prob=0.5) * 2. - 1.
        term1 = tf.matmul(X, params['mu'][0]) + params['b'][0]
        R = tf.nn.dropout(tf.ones_like(term1), keep_prob=0.5) * 2. - 1.
        act = term1 + tf.matmul(X * S,  tf.sqrt(params['sigma'][0])) * R
        hidden_states.append( tf.nn.relu( act ) )

    layer_counter = len(params['mu']) - 1
    for mu, sigma, b in zip(params['mu'][1:], params['sigma'][1:], params['b'][1:]):
        layer_counter -= 1
        for h_idx, h in enumerate(hidden_states):
            S = tf.nn.dropout(tf.ones_like(h), keep_prob=0.5) * 2. - 1.
            term1 = tf.matmul(h, mu) + b
            R = tf.nn.dropout(tf.ones_like(term1), keep_prob=0.5) * 2. - 1.
            act = term1 + tf.matmul(h * S,  tf.sqrt(sigma)) * R
            if layer_counter != 0:
                hidden_states[h_idx] = tf.nn.relu( act ) + hidden_states[h_idx]
            else:
                hidden_states[h_idx] = act

    return hidden_states


def fprop_bayes_nn(X, params, n_samples=10):
# batch_size = tf.shape(X)[0]                                                                                                                                                                                                                               
    hidden_states = []
    # take multiple samples at first layer                                                                                                                                                                                                                  
    for s_idx in range(n_samples):
        S = tf.nn.dropout(tf.ones_like(X), keep_prob=0.5) * 2. - 1.
        term1 = tf.matmul(X, params['mu'][0]) + params['b'][0]
        R = tf.nn.dropout(tf.ones_like(term1), keep_prob=0.5) * 2. - 1.
        act = term1 + tf.matmul(X * S,  tf.sqrt(params['sigma'][0])) * R
        hidden_states.append( tf.nn.relu( act ) )

    layer_counter = len(params['mu']) - 1
    for mu, sigma, b in zip(params['mu'][1:], params['sigma'][1:], params['b'][1:]):
        layer_counter -= 1
        for h_idx, h in enumerate(hidden_states):
            S = tf.nn.dropout(tf.ones_like(h), keep_prob=0.5) * 2. - 1.
            term1 = tf.matmul(h, mu) + b
            R = tf.nn.dropout(tf.ones_like(term1), keep_prob=0.5) * 2. - 1.
            act = term1 + tf.matmul(h * S,  tf.sqrt(sigma)) * R
            if layer_counter != 0:
                hidden_states[h_idx] = tf.nn.relu( act ) 
            else:
                hidden_states[h_idx] = act

    return hidden_states


def fprop_dropout_nn(X, params, n_samples=10):
    # batch_size = tf.shape(X)[0]                                                                                                                                                 
    hidden_states = []
    # take multiple samples at first layer                                                                                                                                         
    for s_idx in range(n_samples):
        X_dropped = tf.nn.dropout(X, keep_prob=1 - 0.005)                                                                                                                   
        act = tf.matmul(X_dropped, params['w'][0]) + params['b'][0]                                                                                                                      
        hidden_states.append( tf.nn.relu( act ) )

    layer_counter = len(params['w']) - 1
    for w, b in zip(params['w'][1:], params['b'][1:]):
        layer_counter -= 1
        for h_idx, h in enumerate(hidden_states):
            h_dropped = tf.nn.dropout(h, keep_prob=1 - 0.005)
            act = tf.matmul(h_dropped, w) + b
            if layer_counter != 0:
                hidden_states[h_idx] = tf.nn.relu(act) 
            else:
                hidden_states[h_idx] = act

    return hidden_states


def compute_gauss_nll(y, y_pred_linear, noise_prec = .75): 
    return tf.reduce_sum(.5 * noise_prec * tf.square(y - y_pred_linear) + .5 * tf.log(2.*np.pi*(1/noise_prec)), reduction_indices=1, keep_dims=True)


def compute_bernoulli_nll(y, y_pred_linear):
    return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(y_pred_linear, y), reduction_indices=1, keep_dims=True) 


def gauss2gauss_KLD(mu_post, sigma_post, mu_prior, sigma_prior, eps=1e-8):
    # mu_post = tf.Print(mu_post, [mu_post], "Posterior mu: ")
    # sigma_post = tf.Print(sigma_post, [sigma_post], "Posterior sigma: ")
    tf.Assert(tf.greater(tf.reduce_min(sigma_post), 0.), [sigma_post])
    tf.Assert(tf.greater(tf.reduce_min(sigma_prior), 0.), [sigma_prior])
    d = (mu_post - mu_prior)
    temp = tf.square(d) + tf.square(sigma_post)
    #temp = tf.Print(temp, [temp], "Numberator...")
    temp1 =  tf.log(sigma_prior + eps) - tf.log(sigma_post + eps)
    #temp1 = tf.Print(temp1, [temp1], "log ratio of sigmas")
    temp3 = 2*tf.square(sigma_prior)
    #temp3 = tf.Print(temp3, [temp3], "denominator..")

    return temp / temp3 + temp1 - .5


def ard_scale_update(prior_id, prior_params, mu, std):
    param_count = mu.get_shape().as_list()[1]
    moments = tf.reduce_sum(mu*mu + std*std, reduction_indices=1, keep_dims=True)
    
    if prior_id == "inv_gamma":
        prior_opt_scale = tf.sqrt((2. * prior_params['param2'] + moments) /  (param_count + 2. * prior_params['param1'] + 2.))

    elif prior_id == "half_cauchy":
        prior_opt_scale = tf.sqrt((moments - prior_params['param1']**2 * param_count + tf.sqrt(moments**2 + (2*param_count+8) * prior_params['param1']**2 * moments + prior_params['param1']**4 * param_count**2)) / (2*param_count + 4))

    # Log-uniform implemented by just passing alpha=beta=0 to inv Gamma
    
    return prior_opt_scale * tf.ones_like(std)


def add_scale_update(prior_id, prior_params, mu, std):
    param_count = mu.get_shape().as_list()[0] * mu.get_shape().as_list()[1]
    moments = tf.reduce_sum(mu**2 + std**2)

    if prior_id== "inv_gamma":
        prior_opt_scale = tf.sqrt((2. * prior_params['param2'] + moments) /  (param_count + 2. * prior_params['param1'] + 2.))

    elif prior_id == "half_cauchy":
        prior_opt_scale = tf.sqrt((moments - prior_params['param1']**2 * param_count + tf.sqrt(moments**2 + (2*param_count+8.) * prior_params['param1']**2 * moments + prior_params['param1']**4 * param_count**2)) /(2*param_count + 4))

    return prior_opt_scale * tf.ones_like(std)


def ard_add_scale_update(prior_id, prior_params, old_tau, mu, std):
    param_count_cols = mu.get_shape().as_list()[1]
    param_count_rows = mu.get_shape().as_list()[0]
    moments = tf.reduce_sum(mu**2 + std**2, reduction_indices=1, keep_dims=True)

    if prior_id == "inv_gamma":
        opt_lambda = tf.sqrt( (2. * prior_params['param2'] * old_tau + moments) / (old_tau * 2. * prior_params['param1'] + old_tau * param_count_cols + old_tau * 2))            
        opt_tau = tf.sqrt( (2. * prior_params['param2'] + tf.reduce_sum((1./opt_lambda) * moments)) / (param_count_cols * param_count_rows + 2 + 2 * prior_params['param1']) )       

    if prior_id == "half_cauchy":
        moments_tau = (1./old_tau) * moments
        opt_lambda = tf.sqrt((moments_tau - prior_params['param1']**2 * param_count_cols + tf.sqrt(moments_tau**2 + (2*param_count_cols+8) * prior_params['param1']**2 * moments_tau + prior_params['param1']**4 * param_count_cols**2)) / (2*param_count_cols + 4))
                
        moments_lamb = tf.reduce_sum((1./opt_lambda) * moments)
        param_count = param_count_rows * param_count_cols
        opt_tau = tf.sqrt((moments_lamb - prior_params['param1']**2 * param_count + tf.sqrt(moments_lamb**2 + (2*param_count+8) * prior_params['param1']**2 * moments_lamb + prior_params['param1']**4 * param_count**2)) / (2*param_count + 4))

    # Log-uniform implemented by just passing alpha=beta=0 to inv Gamma   

    return opt_lambda * opt_tau * tf.ones_like(std)
