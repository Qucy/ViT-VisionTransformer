# PVT - Pyramid Vision Transformer

### 1. Introduction

There are 2 papers for Pyramid Vision Transformer

- Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions
- PVTv2: Improved Baselines with Pyramid Vision Transformer

##### 1.1 Abstraction

- Different from ViT that typically has low-resolution outputs and high computational and memory cost, PVT can be not only trained on dense partitions of the image to achieve high output resolution, which is important for dense predictions but also using a progressive shrinking pyramid to reduce computations of large feature maps.
- PVT inherits the advantages from both CNN and Transformer, making it a unified backbone in various vision tasks without convolutions by simply replacing CNN backbones.
- Validate PVT by conducting extensive experiments, showing that it boosts the performance of many downstream tasks, e.g., object detection, semantic, and instance segmentation. For example, with a comparable number of parameters, RetinaNet+PVT achieves 40.4 AP on the COCO dataset, surpassing RetinNet+ResNet50 (36.3 AP) by 4.1 absolute AP. We hope PVT could serve as an alternative and useful backbone for pixel-level predictions and facilitate future researches.

##### 1.2 Research background

ViT has shown a great performance in image classification, but due to it's using vanilla transformer architecture, it will consume lots of GPU memory during training, a single 224x224 image can consume around 300M memory. Makes it hard to apply to other computer vision tasks like object detection, instance segmentation and instance detection.

##### 1.3 Research result

PVT achieved a better performance in many tasks including image classification, object detection and semantic segmentation. Improved 2~3% in most of tasks.

![pvt1_result](https://github.com/Qucy/VisionTransformer/blob/master/img/pvt1_result.jpg)

##### 1.4 Research meaning

To show that transformer can be applied in many downstream computer vision tasks.



### 2. PVC architecture

Below image depicting the overall architecture of PVC network.

![pvt_overall](https://github.com/Qucy/VisionTransformer/blob/master/img/pvt_overall.jpg)

##### 2.1 pyramid architecture

Below image depicting the traditional architecture for CNNs(a), Vision Transformer(b) and Pyramid Vision Transformer(c).  PVT is very similar with CNNs, both reducing image width, height while increase channels layer by layer, or we can call it pyramid network But why doing this, any reason behind this ?

![pvt_a1](https://github.com/Qucy/VisionTransformer/blob/master/img/pvt_a1.jpg)

One reason is that the different image need different patch size. A simple image may just need 4 patches but a complicate image may need 9 or 16 patches to make a good prediction (reference: https://arxiv.org/abs/2105.15075).  And in may downstream CV tasks, we need to split image to very small patches, in segmentation tasks the prediction is at pixel level.  So we want to this pyramid architecture to help us to find different objects in terms of difficulty or size.

![downstream_tasks](https://github.com/Qucy/VisionTransformer/blob/master/img/downstream_tasks.jpg)

Today the most widely used pyramid network architecture is FPN(d). But can we use FPN + ViT to generate a network directly ?

![fpn](https://github.com/Qucy/VisionTransformer/blob/master/img/fpn.jpg)

Before answer this question we need to take a look at attention's calculation.

- To calculate similarity (Q, K) the complexity is O(n^2)
- To generate Q,K,V complexity is O(n)
- Attention map in memory is O(n^2)

Number of Q, K is depending on patch size, if patch size is small, then will have more Q,K,  more time to calculate MHA and more memory need in GPU. If patch size is big, the feature map aspect ratio will be bigger, will impact on model performance. Hence in PVT they use a technical called Spatial Reduction Attention to solve this problem.

##### 2.2 Spatial Reduction Attention

Below image describe what is Spatial Reduction Attention. Basically reduce the K and V before the attention operation. The reason don't reduce Q is referring to the patch size and we don't want to change the patch size which may have impact on performance.

![sra](https://github.com/Qucy/VisionTransformer/blob/master/img/sra.jpg)

### 3. Experiments Results

##### 3.1 image classification

PVT has the similar parameters and GFLOPs compared with baseline models but have 2~5% performance improvement.

![result1](https://github.com/Qucy/VisionTransformer/blob/master/img/result1.jpg)

##### 3.2 Object detection

After backbone network change from ResNet to PVT all the model has an obvious improvement.

![result2](https://github.com/Qucy/VisionTransformer/blob/master/img/result2.jpg)

##### 3.3 Semantic Segmentation

 After backbone network change from ResNet to PVT all the model has an obvious improvement.

![result3](https://github.com/Qucy/VisionTransformer/blob/master/img/result3.jpg)



### 4. Summary

- Understand why use FPN in network
- Understand MHA performance problem
- Understand SRA and how it reduce calculation and memory consumption
- PVT - a vision transformer can used in downstream CV tasks



### 5. PVT-V2

coming soon

