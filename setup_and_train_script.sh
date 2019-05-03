# Build directory structure for training artifacts
mkdir yacht
mkdir yacht/train_logs
mkdir yacht/params

# Make Yacht data set splits and pickle them
python yacht_data/make_splits_and_pkl.py

# Train Tail-Adaptive MC Dropout Network
python train_dropout_net.py --hidden_layers 1 

# Train ARD ResNet
python train_bnn.py --hidden_layers 2 --prior_struct ard --hyperprior_dist inv_gamma --hyperprior_param1 3. --hyperprior_param2 3. 

# Train ADD ResNet
python train_bnn.py --hidden_layers 2 --prior_struct add --hyperprior_dist inv_gamma --hyperprior_param1 3. --hyperprior_param2 3. 

# Train ARD-ADD ResNet
python train_bnn.py --hidden_layers 2 --prior_struct ard-add --hyperprior_dist inv_gamma --hyperprior_param1 3. --hyperprior_param2 3.