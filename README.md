# Fixed Separable convolutions in CNNs
In this work, I replace standard convolutions with fixed separable convolutions in several popular neural networks. I study how these modifications influence performance of CNNs. Several different domains and networks are tried here.
1) Image Classification with ResNets  
2) Image Generation with DCGAN  

**Note:** Some of the code used here is taken from open source repositories or is influenced by work of other people. 
Complete list of references and credits will be added later.
The main sources are:
- https://github.com/akamaster/pytorch_resnet_cifar10
- https://github.com/hysts/pytorch_image_classification
- https://github.com/pytorch/examples/tree/master/dcgan


## What's inside

1. Tensorboard statistics is under the `runs` folder. These can be run with `tensorboard --logdir=runs`
2. The final state for each model can be found in `model_saves`.
3. `models/nets` contains fully built nets, `models/modules` contains building blocks for these nets.

## Dependencies
The code is written in python and uses pytorch.  
Dependencies can be installed via:
```
conda env create -f  environment.yml
```

## Licence

All files are provided under the terms of the Apache License, Version 2.0.
