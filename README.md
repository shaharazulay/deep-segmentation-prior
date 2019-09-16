# Deep Segmentation Prior
## Unsupervised Image Segmentation With Deep Segmentation Prior
***

![alt text](https://github.com/shaharazulay/deep-segmentation-prior/blob/master/docs/figures/prior_over_epochs.png)

### Abstract
***
We investigate the use of Deep Image Prior (DIP) for unsupervised image segmentation. As in the case of supervised image segmentation, the algorithm output is an assignment of label for each pixel. However, unlike supervised segmentation, no training images or ground truth labels of pixels are available for learning. Therefore, we use DIP with low iteration volume as a segmentation prior for well known unsupervised segmentation algorithms such as K-Means (Lloyd, 1982) and also a deep learning algorithm of (Kanezaki, 2018). We show that this prior can lead to promising results for foreground-background separation as well as multi-class segmentation. 

### Introduction
***
Image segmentation has long been a core vision problem and the focus of many research efforts. Currently, under the heavily researched supervised setting, deep convolutional networks (CNNs) are the state-of-the-art for this task. The unsupervised settings of this problem is of special interest due to the inherit difficulty in acquiring high quality labeled data representing the different object classes in a given image. The segmentation of an image can vary from a simpler settings as foreground-background segmentation to more advanced settings such as multi-class segmentation where the number of classes is not known in advance.

In light of the challenges that rise from unsupervised segmentation, we wish to mitigate them by using a ”smart” (yet) simple prior. (Ulyanov et al., 2018) were first to set the terminology Deep Image Prior (DIP), when they showed that a great deal of image statistics are captured by the structure of a convolutional image generator independent of the learning process itself. The DIP network was shown as sufficient to capture the low-level statistics of a single image, without prior knowledge or supervision. Influenced by their work, we wish to set the terminology ”Deep Segmentation Prior” (DSP), a variant of the DIP network with low iteration volume, that captures the attributes inside an image that are helpful for finding its segmentation.

We show that unsupervised clustering over the DSP achieves promising results for the task of foreground-background seg- mentation. We also show, that the DSP can serve as a powerful starting point the more complex task of multi-class image segmentation, helping in preventing over-segmentation and reducing training time.


### Results
***
See our full [paper](https://github.com/shaharazulay/deep-segmentation-prior/blob/master/docs/Unsupervised_Image_Segmentation_With_DSP.pdf) for more details.

![alt text](https://github.com/shaharazulay/deep-segmentation-prior/blob/master/docs/figures/prior_clustering.png)
