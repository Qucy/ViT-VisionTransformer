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

coming soon





 
