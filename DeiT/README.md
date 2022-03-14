### DeiT - Training data-efficient image transformers & distilling through attention



##### 1. Abstract

Recently, neural networks purely based on attention were shown to address image understanding tasks such as image classification. These high performing vision transformers are pre-trained with hundreds of millions of images using a large infrastructure, thereby limiting their adoption. In this work, we produce competitive convolution-free transformers by training on ImageNet only. We train them on a single computer in less than 3 days. Our reference vision transformer (86M parameters) achieves top-1 accuracy of 83.1% (single-crop) on ImageNet with no external data. More importantly, we introduce a teacher-student strategy specific to transformers. It relies on a distillation token ensuring that the student learns from the teacher through attention. We show the interest of this token-based distillation, especially when using a convnet as a teacher. This leads us to report results competitive with convnets for both ImageNet (where we obtain up to 85.2% accuracy) and when transferring to other tasks. We share our code and models.

##### 2. Research background

The research background is although ViT show a great performance in different CV tasks, but it needs a lot of data to train and has too many parameters which is hard to apply to downstream tasks. Below images shows that when the data size keep increasing the performance for large model is also increasing.

![ViT_data](https://github.com/Qucy/ViT-VisionTransformer/blob/master/img/ViT_data.jpg)

Before we start to look into DeiT, let's stop and think why Transformers need so many data. It's all because of the inductive bias. For time series problem we think there is time sequence dependency between each element so we designed RNN to capture this kind of information. For image problem we think there is a space relationship between the neighbor pixels so that's why we design the CNN to capture the space information. But for transformers there is no inductive bias and prior knowledge, so naturally it needs more data to find out those hidden rules.

##### 3. DeiT vs ViT vs Efficient Net

Below image shows the performance for DeiT model and how many image can handled per seconds. And research meaning for DeiT is:

- define a standard procedure for training Vision Transformers, without Google JFT dataset, with less training time
- become a baseline model for the latter vision transformer models

![deit_performance](https://github.com/Qucy/ViT-VisionTransformer/blob/master/img/deit_performance.jpg)

##### 4. Method

DeiT has the same network architecture compared with ViT but why DeiT can achieve a such good performance even without JFT datasets. It can be summaries as below:

- More data argumentation, it uses lots of data argumentation in training
- Adjust hyper parameter accordingly, when using lots data argumentation, hyper parameter also need to be tuned accordingly
- Knowledge distilling

Below image depicting the training difference ViT and DeiT

![DeiT_hp](https://github.com/Qucy/ViT-VisionTransformer/blob/master/img/DeiT_hp.jpg)

Below image depicting the performance matrix for different data argumentations or other tricks.

![DeiT_test](https://github.com/Qucy/ViT-VisionTransformer/blob/master/img/DeiT_test.jpg)

For knowledge distilling it adds a teacher loss compared during training. So the total loss equals to soft loss + hard loss.

- soft loss = KL loss(softmax(student), softmax(teacher))
- hard loss = cross entropy loss(prediction, label)

Below image depicting the knowledge distilling for DeiT

![DeiT_distilling](https://github.com/Qucy/ViT-VisionTransformer/blob/master/img/DeiT_distilling.jpg)

##### 4. Performance

![DeiT_performance1](https://github.com/Qucy/ViT-VisionTransformer/blob/master/img/DeiT_performance1.jpg)