### Predictive Coding in Python

## <img alt="logo" src="https://www.frontiersin.org/files/Articles/18458/fpsyg-02-00395-r3/image_m/fpsyg-02-00395-g003.jpg" height="180"> 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

A `Python` implementation of _An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic Plasticity_

[[Paper](https://www.mrcbndu.ox.ac.uk/sites/default/files/pdf_files/Whittington%20Bogacz%202017_Neural%20Comput.pdf)]

Based on the `MATLAB` [implementation](https://github.com/djcrw/Supervised-Predictive-Coding) from [`@djcrw`]

## Requirements
- `numpy`
- `torch`
- `torchvision` 


## Tasks
- Include model from _A tutorial on the free-energy framework for modelling perception and learning_
- Add additional optimisers
- Measure number of iterations
- The initial space of mu needs to be sufficently large - ensembles of amortised weights or slow learning rate?
- Test pure PC accuracy
