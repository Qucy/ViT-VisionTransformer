# ViT - Vision Transformer

### 1. Introduction

The paper is named 'An Image is Worth 16x16 words: Transformers for Image Recognition as Scale'  and published on ICLR2021 by Google Brain team. 

##### 1.1 Abstraction

- Transformer has become the de-facto standard for natural language processing tasks, but not widely used in computer vision
- In computer vision attention is is more like a supplement or enhancement
- ViT achieves a good performance on image classification tasks
- After trained on large datasets ViT can achieve a comparable performance with SOTA CNN

##### 1.2 Research background

Transformer achieved a huge successful in NLP but still not widely used in computer vision, this paper focus on whether a transformer can be applied in CV successfully. And transformer has below 3 major advantages:

- parallel computation
- global attention
- easily stack on top of each other

##### 1.3 Research result

ViT achieve a comparable performance with ResNet, below image depicting accuracy on different dataset.

![vit_results](https://github.com/Qucy/ViT-VisionTransformer\img\vit_results.jpg)

- datasets: ImageNet, CIFAR, Oxford and Google private datasets JFT(20~30 time compared with ImageNet)
- ViT-H/14: ViT-Huge model, input sequence is 14x14

##### 1.4 Research meaning

- show that pure Transformer can be used in Computer Vision and achieve a good performance
- opening the Transformer research in Computer Vision



### 2. ViT architecture 

ViT overall architecture is depicting as below, it following the original Transformer and only using encoder. For input image it will split to N x N small patches and flatten the patches, add with position embedding then input the Transformer Encoder. Finally it will use zero position embedding to make classification prediction.

![vit](https://github.com/Qucy/ViT-VisionTransformer/blob/master/img/convvit.jpg)



##### 2.1 Patch + Position Embedding

Patch + Position Embedding is mainly want to split the image into small patches and then give a position number to each patches to keep the sequence information. But this will lose the space information, like the patch 1 image is on top of patch 4 image.

![pos_embedding](https://github.com/Qucy/ViT-VisionTransformer/blob/master/img/convpos_embedding.jpg)

So how to we split one image into small patches ? Here in ViT we use conv layers with specified strides then we can split one image into small patches. For example if we use a 16x16 kernel with 16x16 strides, so the cone layers will extract feature maps every 16 pixels and no overlap between each feature maps. If our input image is 224 x 224 x 3, with a 16 x 16 x 768 filters we can extract a 14 x 14 x 768 feature map.

![conv](https://github.com/Qucy/ViT-VisionTransformer/blob/master/img/convconv.gif)

After we split one image into small patches, then we going flatten width and height, for example if our feature map is 14 x 14 x 768 then we going to resize it to 196 x 768.  After this we going to create a 1x768 CLS token which is **0*** in the below image. Then we going to concat our flattened feature maps and CLS token to generate a 197 x 768's feature map. After adding the CLS token we need to add position information into this feature map. We generate a 197 x 768's trainable matrix and add with feature map. 

![pos_embedding](https://github.com/Qucy/ViT-VisionTransformer/blob/master/img/convpos_embedding.jpg)

Source code is as below

....

##### 2.2 Transformer encoder





### 3. ViT

coming soon









 
