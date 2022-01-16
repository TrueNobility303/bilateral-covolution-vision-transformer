# bilateral-covolution-vision-transformer

## Abstract

* We design a novel network BCVT(bilateral-covolution-vision-transformer), which improves CVT(covolution-vision-transformer) in Paper [CvT: Introducing Convolutions to Vision Transformers](https://arxiv.org/pdf/2103.15808v1.pdf) for image classification task on CIFAR10. Our inspiration is from BiSENET, which is psotulated in Paper [BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation](https://arxiv.org/pdf/1808.00897v1.pdf). The improved model(BCVT) consists of bilateral paths, outperforming the original network(CVT) in the same circumstances.
* We also compare BCVT with ResNet50 in the circumstances of both same parameters and same flops. BCVT is much more powerful can ResNet50, which is a pure convolution networks(CNN) without any transformer modules.
* We provide ablation studies on position embedding， covolution embedding and other essential parts in BCVT.
* We use optimizer SAM from Paper [Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/pdf/2010.01412v3.pdf) for better performance.
* Refer to out report for more details!

Here is another [link](https://github.com/caiyancheng/Computer-Vision-Final-project) with similar contents.

## 摘要（中文版本）

* 设计网络BCVT(bilateral-covolution-vision-transformer),改进了CVT进行CIAFR10图像分类，改进灵感来自BiSENet，改进后的模型具有双通道，并且可以获取比原网络更好的效果
* 对比了同paramters,同flops的BCVT和ResNet50之间的区别，BCVT也具有更好的效果
* 可进行position embedding， covolution embedding，等transformer相关的探究实验
* 使用优化器SAM获得更好的效果


## Training Curve

![acc](https://github.com/TrueNobility303/bi-covolution-vision-transformer/edit/master/curve.png)


