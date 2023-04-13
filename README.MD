![Evolly logo](docs/images/logo.png)

**Evolly** is an [evolutionary](https://en.wikipedia.org/wiki/Evolutionary_algorithm) 
[neural architecture search](https://en.wikipedia.org/wiki/Neural_architecture_search) 
framework designed to make running evolution as flexible as possible.

Learn details of the Evolly's pipeline in our [Towards Data Science post](https://medium.com/@revisorteam).

## What does Evolly do?

---

Evolly allows you to:
1. **Boost metrics of your deep learning model** by tuning backbone 
architecture
2. **Search for new backbone architectures** by finding optimal types, 
order of the blocks and optimizing block parameters (kernel sizes, strides, filters and dropouts).

You can apply it with to any Deep Learning task:
classification, detection, segmentation, pose estimation, GAN, etc. 

![Evolution visualization](docs/images/evolution_visualization1.gif)

## Features

---

We've added following features to make it possible to implement Evolly 
in any training pipeline:

* Build models using common DL frameworks (tensorflow, torch)
* Set multiple branches (stems) of different data types
* Define custom backbone depth and width
* Pass custom architecture blocks
* Choose parameters to mutate
* Customize allowed values and intervals of the mutations
* Run training in distributed or parallel mode
* Monitor evolution via TensorBoard
* Estimate search space size
* Visualize evolution

![Evolution visualization](docs/images/evolution_visualization2.gif)

## Getting started

---
To launch evolution with Evolly:

1. Make sure you have tensorflow >= 2.3 and torch >= 1.9.0 installed
2. Install Evolly via pip: ``pip install evolly``
3. Follow [Making your first evolution](GETTING_STARTED.MD) guide


## Improvements

---

We are open to any help. Check out [our ideas here](https://medium.com/@revisorteam) to learn how we can upgrade Evolly together:

- [ ] Test default PyTorch blocks
- [ ] Add new data types
- [ ] Add new default blocks
- [ ] Utilize [mutation rate](https://en.wikipedia.org/wiki/Mutation_rate) and add mutation probabilities
- [ ] Implement reinforcement learning
- [ ] Upgrade branch connections
- [ ] Implement ability to build multiple branches with torch


### References

---

* [EvoPose2D](https://www.researchgate.net/publication/355101183_EvoPose2D_Pushing_the_Boundaries_of_2D_Human_Pose_Estimation_Using_Accelerated_Neuroevolution_With_Weight_Transfer): genotype storing approach and MobileNetV2 block implementation
* [Inception_ResNet_v2](https://github.com/Sakib1263/Inception-InceptionResNet-SEInception-SEInceptionResNet-1D-2D-Tensorflow-Keras) block implementation
* [ResNet](https://github.com/keras-team/keras/blob/master/keras/applications/resnet.py) block implementation

### Contacts

---

Contact us if you are interested in collaborating or ready to invest 
in us: revisorteam@pm.me