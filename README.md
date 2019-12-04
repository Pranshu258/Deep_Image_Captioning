# Deep RL based Image Captioning with Embedding Reward
Pranshu Gupta, Deep Learning @ Georgia Institute of Technology

### Introduction
   Image Captioning is the task of assigning a short textual description to an image which captures the semantic meaning of the visual artifacts. Various approaches have been proposed to generate image captions - encoder decoder models being the most common. New techniques have also emerged recently in which reinforcement learning methods are employed on top of state of the art models to achieve better results. In this article, we discuss a few of these approaches as proposed in the paper "Deep reinforcement learning-based captioning with embedding reward" by Ren et. al.

### Model Architecture
We define the agent as the caption generator, the state as the visual features and caption generated so far, the action space as the available vocabulary, and the reward as the similarity between the visual and text embedding of the image and the ground truth captions in the same vector space. The task is defined as choosing the next word for the caption (action) given the visual (image) and semantic (caption so far) features as state, as per the approximate optimal policy. 

The policy, reward and the value functions are approximated with deep neural network, with visual features encoded with the help of CNN based network (VGG-16) and semantic features with the help of RNN based networks. 

### Implementation
The component networks, their training and evaluation code is implemented in the Deep_Captioning.ipynb notebook. MSCOCO dataset is used for all the training and evaluation with the 2014 splits. The implementation works with 512 dimensional features vectors instead of raw images. These feature vectors were extracted from the fully connected (fc7) layer of VGG-16 which gives 4096 dimensional vectors, the PCA was applied on them to get 512 dim vectors. The notebook itself is self contained with code and description.