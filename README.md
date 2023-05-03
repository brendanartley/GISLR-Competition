# Google-GISLR-Competition

This repository contains code for the 54th place solution in the [Google - Isolated Sign Language Recognition competition](https://www.kaggle.com/competitions/asl-signs) on Kaggle.

The final model was an ensemble of 2 transformers trained on different seeds.

## Improvenments

Stochastic Weight Averaging with cyclical learning rate for more generalizable models (ex. apply after 10 epochs).
- Averaging Weights Leads to Wider Optima and Better Generalization [Paper](https://arxiv.org/abs/1803.05407
- Good article by Max Pechyonkin explaining this idea [here](https://pechyonkin.me/stochastic-weight-averaging/))
- [Tensorflow Implementation](https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/SWA)
- Solution that used this idea: [Ohkawa3's 4th Place](https://www.kaggle.com/competitions/asl-signs/discussion/406673)

Spend time Ensembling NNs on local CV
- Only did this on the last day of the competition
- If you can decrease the size of the NN without degrading CV then it seems like this is an easy way to boost performance

DepthwiseConv1D, DepthwiseConv2D seemed to be a good learnable layer that other competitors used to "smooth" the data
- https://www.tensorflow.org/api_docs/python/tf/keras/layers/DepthwiseConv1D
- Used in many top solutions: [Forrato's 2nd place](https://www.kaggle.com/competitions/asl-signs/discussion/406306), [Ruslan Grimov's 3rd place](https://www.kaggle.com/competitions/asl-signs/discussion/406568)

Some good points from [Chris Deotte's 44th place discussion](https://www.kaggle.com/competitions/asl-signs/discussion/406302#2244217)
Test different batch sizes
- "Many Kagglers overlook the fact that changing batch size can make a big difference for models. We should always try 0.25x, 0.5x, 2x, 4x batch size and change the learning rate for those experiments to be 0.25x, 0.5x, 2x, 4x respectively."
- "NN always benefits from more data and data augmentation."
- "The easiest way to boost CV LB for NN is to ensemble NN"
