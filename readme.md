# Overview
Fine-tuning of a pretrained CNN. For this project, the VGG11 architecture was chosen for all experiments.

# Data preprocessing 
The mean and standard deviation for each color channel (RGB) across all pixels and images in the training set used for normalization.
- **Mean**: tensor([0.4631, 0.4483, 0.3237])
- **Standard Deviation**: tensor([0.2830, 0.2650, 0.2754])

# Freezed parameters
All parameters were freezed and then deleted classifier part
that maps features scores of 1000 Imagenet classes. 
Then a new module for 10 classes was added and replaced last linear layer in the last classifier. 

For purpose of finding a suitable learning rate SGD optimizer and hyperparameter 𝑚𝑜𝑚𝑒𝑛𝑡𝑢𝑚 = 0.9 was used. Cross Entropy loss was used  as a loss function. 
Then  the other learning rate order was determined by experimentations with 𝑙𝑒𝑎𝑟𝑛𝑖𝑛𝑔𝑅𝑎𝑡𝑒𝑠 = [0.1, 0.01, 0.001, 0.0001].

# Not freezed parameters