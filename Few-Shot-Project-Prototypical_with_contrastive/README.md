# Few-Shot-Project

 - Run the train.py file for training and testing purposes. 
 - parser.py has the hyperparameters that can be modified. 
 - resnet_simclr.py file has the model for mini-imagenet dataset while resnet_simclr_omni.py has the model for omniglot dataset
(omniglot is 1-D while mini-imagenet is 3-D input, hence first layer of the model is different).
 - This branch uses the model weights obtained from resnet_simclr branch code and then trains a prototypical network.
