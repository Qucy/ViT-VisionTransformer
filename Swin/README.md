### Swin Transformer - Hierarchical Vision Transformer using Shifted Windows



##### 1. Abstraction

The paper is published on ICCV 2021 and receive best paper award. The paper introduced a new vision transformer called Swin Transformer. Basically it try to solve 2 problems, 1) how big the token should be for computer vision task, 2) how many tokens should be used for computer vision tasks. So they proposed a hierarchical vision transformer whose representation is computed with Shifted windows to solve above 2 problems.

![swin_vs_vit](https://github.com/Qucy/ViT-VisionTransformer/blob/master/img/swin_vs_vit.jpg)



##### 2. Research background

For NLP we know that most of the time the token is just a word or couple of words, but for CV how big the token should be, a 16x16 pixels image or 32x32 by pixels image ? Well that depends,  how complicate image is and how big is object in that image. Below images demonstrates that the different image need different patches to make best prediction.

![patch1](https://github.com/Qucy/ViT-VisionTransformer/blob/master/img/patch1.jpg)

Even same object may have different size in different image, depends on how far away from the camera. For some irregular objects like cloud , sky and water it's hard to use any fixed tokens.

![cars](https://github.com/Qucy/ViT-VisionTransformer/blob/master/img/cars.jpg)

Another thing is that, ViT will generate too many tokens and need too many memories when calculating  attentions and can not used in many downstream tasks, especially those tasks need to handling high resolution images. For example, a 224x224 image with 4x down sampling, will generate 56 * 56 = 3136 tokens. This is a lot and this is just for 1 image only. So here brings a dilemma as below:

- using small patches -> too many tokens -> too many memory and computation -> out of memory,can not train
- using big patches -> less tokens -> low resolution feature maps -> performance decay issue



##### 3. Research Meaning

- This is the first pure vision transformer can be widely used in downstream tasks

- The most popular vision transformer so far, 500+ references till Dec 2021

  ![swin_ref](https://github.com/Qucy/ViT-VisionTransformer/blob/master/img/swin_ref.jpg)



##### 4. Overall architectural

Below image depicting the overall architectural for Swin Transformer. It's have 4 stage same as ViT and PVT, but different with ViT and PVT it use a patch merging layer after each stage, reduce the image height and width while increasing the channels. And the other difference it use a W-MSA and SW-MSA to reduce the computation inside attention module. Let's take a deep look in these 2 differences.

![swin_arc](https://github.com/Qucy/ViT-VisionTransformer/blob/master/img/swin_arc.jpg)

##### 4.1 Patch merging

To produce a hierarchical representation, the number of tokens is reduced by patch merging layers as the network gets deeper. The first patch merging layer concatenates the features of each group of 2 × 2 neighboring patches, and applies a linear layer on the 4C-dimensional concatenated features. This reduces the number of tokens by a multiple of 2×2 = 4 (2× downsampling of resolution), and the output dimension is set to 2C.

Below image demonstrate how patch merging is working.

![patch_merging](https://github.com/Qucy/ViT-VisionTransformer/blob/master/img/patch_merging.jpg)

##### 4.2 W-MSA

We know that the dilemma for MHA is that, we want to reduce the number of Q,K,V but we don't want hurt performance too much. In PVT the author use Spatial Resolution Attention to reduce K and V. The problem for SRA is that Q and K have different receptive field, the complexity for calculate attention is still O(n^2). So can we let Q and K have the same receptive filed and reduce calculation complexity at the same time.

![sra_v2](https://github.com/Qucy/ViT-VisionTransformer/blob/master/img/sra_v2.jpg)

Since we want to calculate attention between each Q and K, so that Q and K can have the same receptive field. Can we only calculate a fixed size Q and K, how about calculate Q and K in a fixed size window. That's exactly what W-MSA or window multiscale self attention is doing . It defines an window an only calculate attention with in this window. Hence for a 64x64 patches with 4 window the calculation reduced to 16x16x4.

![wmsa](https://github.com/Qucy/ViT-VisionTransformer/blob/master/img/wmsa.jpg)

By using W-MSA Q and K will have the same receptive field and reduce the complexity to O(n)

##### 4.3 SW-MSA

Although W-MSA solve 2 problems we mentioned above but it brings an new problem, because we only calculate attention with in a fixed window so we lost the global attention. Hence author propose another attention called shifted window multiscale self attention. The ideal is that we add one more layer using SW-MSA, in SW-MSA we use different window sized to calculate self attention. And if you overlap these 2 layers, the window inside the SW-MSA is kind of like a bridge between 2 windows in the W-MSA layer. Hence it can provide some information between 2 fixed size window in the first layer. In the below image Layer 1 stands for W-MSA layer and Layer 1+1 stands for SW-MSA layer.

![swmsa](https://github.com/Qucy/ViT-VisionTransformer/blob/master/img/swmsa.jpg)

But here brings another problem, is that we can not use the same code to calculate self attention for SW-MSA because each window may have different size.  Hence in this paper they use a method called **cyclic shift** to handle this problem. 

- First assign each window an index, then roll the window and make it can be divide equally like W-MSA.
- Calculate raw attention for each divided window
- Calculate mask matrix
- raw attention @ mask = final attention map

![cyclic_shift](https://github.com/Qucy/ViT-VisionTransformer/blob/master/img/cyclic_shift.jpg)



##### 5. Experiments and results

##### 5.1 classification

Swin has a better performance compared with other model if the GFLOPs is almost the same.

![swim-classification](https://github.com/Qucy/ViT-VisionTransformer/blob/master/img/swim-classification.jpg)

##### 5.2 Object detection

![swin_objection_detection](https://github.com/Qucy/ViT-VisionTransformer/blob/master/img/swin_objection_detection.jpg)

##### 5.3 Semantic Segmentation

![swin_semantic_segmentation](https://github.com/Qucy/ViT-VisionTransformer/blob/master/img/swin_semantic_segmentation.jpg)





