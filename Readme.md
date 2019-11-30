## How to  use this code

This repo requires **tensorflow-gpu-1.5.0** or other compatible version of tensorflow,

you can run it simple by type 

**python cifar10.py**

### Does CNN really need downsampling (upsampling)?

In common convolutional neural networks, sampling is almost ubiquitous, formerly max_pooling, and now strided convolution.
Take the vgg network as an example, which uses quite a lot of max_pooling,

![vgg](https://github.com/akkaze/cnn-without-any-downsampling/blob/master/assets/vgg.png)
<center>The input side is below, you can see that a lot of 2x2 pooling is used in the network</center>
Also, when doing semantic segmentation or object detection, we use quite a lot of upsampling, or transposed convolution.

![fcn](https://github.com/akkaze/cnn-without-any-downsampling/blob/master/assets/fcn.png)
<center>Typical fcn structure, pay attention to the decovolution distinguished by red</center>
Previously, we used fc in the last few layers of the classification network. Later, fc was proved to have too many parameters and poor generalization performance. It was replaced by global average pooling and it first appeared in network in network.

![gap](https://github.com/akkaze/cnn-without-any-downsampling/blob/master/assets/gap.jpg)
<center>GAP aggregates spatial features directly into a scalar</center>
Since then, the paradigm of classification networks behaves like(Relu has been integrated into conv and deconv, without considering any shortcut),

```
Input-->Conv-->DownSample_x_2-->Conv-->DownSample_x_2-->Conv-->DownSample_x_2-->GAP-->Conv1x1-->Softmax-->Output
```

And the paradigm of semantic segmentation network behaves like,

```
Input-->Conv-->DownSample_x_2-->Conv-->DownSample_x_2-->Conv-->DownSample_x_2-->Deconv_x_2-->Deconv_x_2-->Deconv_x_2-->Softmax-->Output
```

However, we have to think about it. Is downsampling and upsampling really necessary? Is it impossible to remove it?

On the classification task of cifar10, I tried to remove the downsampling, change the convolution to a dilated convolution, and the dialation rate increased respectively. The model structure is shown below.

```python
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 32, 32, 16)        448
_________________________________________________________________
batch_normalization (BatchNo (None, 32, 32, 16)        64
_________________________________________________________________
activation (Activation)      (None, 32, 32, 16)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 32, 32, 24)        3480
_________________________________________________________________
batch_normalization_1 (Batch (None, 32, 32, 24)        96
_________________________________________________________________
activation_1 (Activation)    (None, 32, 32, 24)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 32, 32, 32)        6944
_________________________________________________________________
batch_normalization_2 (Batch (None, 32, 32, 32)        128
_________________________________________________________________
activation_2 (Activation)    (None, 32, 32, 32)        0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 32, 32, 48)        13872
_________________________________________________________________
batch_normalization_3 (Batch (None, 32, 32, 48)        192
_________________________________________________________________
activation_3 (Activation)    (None, 32, 32, 48)        0
_________________________________________________________________
global_average_pooling2d (Gl (None, 48)                0
_________________________________________________________________
dense (Dense)                (None, 10)                490
_________________________________________________________________
activation_4 (Activation)    (None, 10)                0
=================================================================
```

After 80 rounds of training, the following classification results were finally obtained,


|   epoch   | loss  |val_accuracy|
|  ----  | ----  |----|
| 10  | 0.9200 |0.6346|
| 20  | 0.7925 |0.6769|
| 30  | 0.7293 |0.7193|
| 40  | 0.6737 |0.7479|
| 50  | 0.6516 |0.7470|
| 60  | 0.6311 |0.7678|
| 70  | 0.6085 |0.7478|
| 80  | 0.5865 |0.7665|

The accuracy curve on validation dataset is shown below,

![acc](https://github.com/akkaze/cnn-without-any-downsampling/blob/master/assets/acc.png)

The final accuracy rate reached 76%. The accuracy rate of a convolutional network with vgg structure with the same parameters is basically around this.

```python
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            [(None, 32, 32, 3)]  0
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 32, 32, 16)   448         input_1[0][0]
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, 32, 32, 16)   64          conv2d[0][0]
__________________________________________________________________________________________________
activation (Activation)         (None, 32, 32, 16)   0           batch_normalization[0][0]
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 32, 32, 16)   2320        activation[0][0]
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 32, 32, 16)   64          conv2d_1[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 32, 32, 16)   0           batch_normalization_1[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 32, 32, 16)   2320        activation_1[0][0]
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 32, 32, 16)   64          conv2d_2[0][0]
__________________________________________________________________________________________________
add (Add)                       (None, 32, 32, 16)   0           activation[0][0]
                                                                 batch_normalization_2[0][0]
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 32, 32, 16)   0           add[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 32, 32, 16)   2320        activation_2[0][0]
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 32, 32, 16)   64          conv2d_3[0][0]
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 32, 32, 16)   0           batch_normalization_3[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 32, 32, 16)   2320        activation_3[0][0]
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 32, 32, 16)   64          conv2d_4[0][0]
__________________________________________________________________________________________________
add_1 (Add)                     (None, 32, 32, 16)   0           activation_2[0][0]
                                                                 batch_normalization_4[0][0]
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 32, 32, 16)   0           add_1[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 32, 32, 16)   2320        activation_4[0][0]
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 32, 32, 16)   64          conv2d_5[0][0]
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 32, 32, 16)   0           batch_normalization_5[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 32, 32, 16)   2320        activation_5[0][0]
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 32, 32, 16)   64          conv2d_6[0][0]
__________________________________________________________________________________________________
add_2 (Add)                     (None, 32, 32, 16)   0           activation_4[0][0]
                                                                 batch_normalization_6[0][0]
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 32, 32, 16)   0           add_2[0][0]
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 32, 32, 16)   2320        activation_6[0][0]
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 32, 32, 16)   64          conv2d_7[0][0]
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 32, 32, 16)   0           batch_normalization_7[0][0]
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 32, 32, 16)   2320        activation_7[0][0]
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 32, 32, 16)   64          conv2d_8[0][0]
__________________________________________________________________________________________________
add_3 (Add)                     (None, 32, 32, 16)   0           activation_6[0][0]
                                                                 batch_normalization_8[0][0]
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 32, 32, 16)   0           add_3[0][0]
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 32, 32, 16)   2320        activation_8[0][0]
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 32, 32, 16)   64          conv2d_9[0][0]
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 32, 32, 16)   0           batch_normalization_9[0][0]
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 32, 32, 16)   2320        activation_9[0][0]
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 32, 32, 16)   64          conv2d_10[0][0]
__________________________________________________________________________________________________
add_4 (Add)                     (None, 32, 32, 16)   0           activation_8[0][0]
                                                                 batch_normalization_10[0][0]
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 32, 32, 16)   0           add_4[0][0]
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 32, 32, 16)   2320        activation_10[0][0]
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 32, 32, 16)   64          conv2d_11[0][0]
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 32, 32, 16)   0           batch_normalization_11[0][0]
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 32, 32, 16)   2320        activation_11[0][0]
__________________________________________________________________________________________________
batch_normalization_12 (BatchNo (None, 32, 32, 16)   64          conv2d_12[0][0]
__________________________________________________________________________________________________
add_5 (Add)                     (None, 32, 32, 16)   0           activation_10[0][0]
                                                                 batch_normalization_12[0][0]
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 32, 32, 16)   0           add_5[0][0]
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 32, 32, 32)   4640        activation_12[0][0]
__________________________________________________________________________________________________
batch_normalization_13 (BatchNo (None, 32, 32, 32)   128         conv2d_13[0][0]
__________________________________________________________________________________________________
activation_13 (Activation)      (None, 32, 32, 32)   0           batch_normalization_13[0][0]
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 32, 32, 32)   9248        activation_13[0][0]
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 32, 32, 32)   4640        activation_12[0][0]
__________________________________________________________________________________________________
batch_normalization_14 (BatchNo (None, 32, 32, 32)   128         conv2d_14[0][0]
__________________________________________________________________________________________________
add_6 (Add)                     (None, 32, 32, 32)   0           conv2d_15[0][0]
                                                                 batch_normalization_14[0][0]
__________________________________________________________________________________________________
activation_14 (Activation)      (None, 32, 32, 32)   0           add_6[0][0]
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 32, 32, 32)   9248        activation_14[0][0]
__________________________________________________________________________________________________
batch_normalization_15 (BatchNo (None, 32, 32, 32)   128         conv2d_16[0][0]
__________________________________________________________________________________________________
activation_15 (Activation)      (None, 32, 32, 32)   0           batch_normalization_15[0][0]
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 32, 32, 32)   9248        activation_15[0][0]
__________________________________________________________________________________________________
batch_normalization_16 (BatchNo (None, 32, 32, 32)   128         conv2d_17[0][0]
__________________________________________________________________________________________________
add_7 (Add)                     (None, 32, 32, 32)   0           activation_14[0][0]
                                                                 batch_normalization_16[0][0]
__________________________________________________________________________________________________
activation_16 (Activation)      (None, 32, 32, 32)   0           add_7[0][0]
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 32, 32, 32)   9248        activation_16[0][0]
__________________________________________________________________________________________________
batch_normalization_17 (BatchNo (None, 32, 32, 32)   128         conv2d_18[0][0]
__________________________________________________________________________________________________
activation_17 (Activation)      (None, 32, 32, 32)   0           batch_normalization_17[0][0]
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, 32, 32, 32)   9248        activation_17[0][0]
__________________________________________________________________________________________________
batch_normalization_18 (BatchNo (None, 32, 32, 32)   128         conv2d_19[0][0]
__________________________________________________________________________________________________
add_8 (Add)                     (None, 32, 32, 32)   0           activation_16[0][0]
                                                                 batch_normalization_18[0][0]
__________________________________________________________________________________________________
activation_18 (Activation)      (None, 32, 32, 32)   0           add_8[0][0]
__________________________________________________________________________________________________
conv2d_20 (Conv2D)              (None, 32, 32, 32)   9248        activation_18[0][0]
__________________________________________________________________________________________________
batch_normalization_19 (BatchNo (None, 32, 32, 32)   128         conv2d_20[0][0]
__________________________________________________________________________________________________
activation_19 (Activation)      (None, 32, 32, 32)   0           batch_normalization_19[0][0]
__________________________________________________________________________________________________
conv2d_21 (Conv2D)              (None, 32, 32, 32)   9248        activation_19[0][0]
__________________________________________________________________________________________________
batch_normalization_20 (BatchNo (None, 32, 32, 32)   128         conv2d_21[0][0]
__________________________________________________________________________________________________
add_9 (Add)                     (None, 32, 32, 32)   0           activation_18[0][0]
                                                                 batch_normalization_20[0][0]
__________________________________________________________________________________________________
activation_20 (Activation)      (None, 32, 32, 32)   0           add_9[0][0]
__________________________________________________________________________________________________
conv2d_22 (Conv2D)              (None, 32, 32, 32)   9248        activation_20[0][0]
__________________________________________________________________________________________________
batch_normalization_21 (BatchNo (None, 32, 32, 32)   128         conv2d_22[0][0]
__________________________________________________________________________________________________
activation_21 (Activation)      (None, 32, 32, 32)   0           batch_normalization_21[0][0]
__________________________________________________________________________________________________
conv2d_23 (Conv2D)              (None, 32, 32, 32)   9248        activation_21[0][0]
__________________________________________________________________________________________________
batch_normalization_22 (BatchNo (None, 32, 32, 32)   128         conv2d_23[0][0]
__________________________________________________________________________________________________
add_10 (Add)                    (None, 32, 32, 32)   0           activation_20[0][0]
                                                                 batch_normalization_22[0][0]
__________________________________________________________________________________________________
activation_22 (Activation)      (None, 32, 32, 32)   0           add_10[0][0]
__________________________________________________________________________________________________
conv2d_24 (Conv2D)              (None, 32, 32, 32)   9248        activation_22[0][0]
__________________________________________________________________________________________________
batch_normalization_23 (BatchNo (None, 32, 32, 32)   128         conv2d_24[0][0]
__________________________________________________________________________________________________
activation_23 (Activation)      (None, 32, 32, 32)   0           batch_normalization_23[0][0]
__________________________________________________________________________________________________
conv2d_25 (Conv2D)              (None, 32, 32, 32)   9248        activation_23[0][0]
__________________________________________________________________________________________________
batch_normalization_24 (BatchNo (None, 32, 32, 32)   128         conv2d_25[0][0]
__________________________________________________________________________________________________
add_11 (Add)                    (None, 32, 32, 32)   0           activation_22[0][0]
                                                                 batch_normalization_24[0][0]
__________________________________________________________________________________________________
activation_24 (Activation)      (None, 32, 32, 32)   0           add_11[0][0]
__________________________________________________________________________________________________
conv2d_26 (Conv2D)              (None, 32, 32, 64)   18496       activation_24[0][0]
__________________________________________________________________________________________________
batch_normalization_25 (BatchNo (None, 32, 32, 64)   256         conv2d_26[0][0]
__________________________________________________________________________________________________
activation_25 (Activation)      (None, 32, 32, 64)   0           batch_normalization_25[0][0]
__________________________________________________________________________________________________
conv2d_27 (Conv2D)              (None, 32, 32, 64)   36928       activation_25[0][0]
__________________________________________________________________________________________________
conv2d_28 (Conv2D)              (None, 32, 32, 64)   18496       activation_24[0][0]
__________________________________________________________________________________________________
batch_normalization_26 (BatchNo (None, 32, 32, 64)   256         conv2d_27[0][0]
__________________________________________________________________________________________________
add_12 (Add)                    (None, 32, 32, 64)   0           conv2d_28[0][0]
                                                                 batch_normalization_26[0][0]
__________________________________________________________________________________________________
activation_26 (Activation)      (None, 32, 32, 64)   0           add_12[0][0]
__________________________________________________________________________________________________
conv2d_29 (Conv2D)              (None, 32, 32, 64)   36928       activation_26[0][0]
__________________________________________________________________________________________________
batch_normalization_27 (BatchNo (None, 32, 32, 64)   256         conv2d_29[0][0]
__________________________________________________________________________________________________
activation_27 (Activation)      (None, 32, 32, 64)   0           batch_normalization_27[0][0]
__________________________________________________________________________________________________
conv2d_30 (Conv2D)              (None, 32, 32, 64)   36928       activation_27[0][0]
__________________________________________________________________________________________________
batch_normalization_28 (BatchNo (None, 32, 32, 64)   256         conv2d_30[0][0]
__________________________________________________________________________________________________
add_13 (Add)                    (None, 32, 32, 64)   0           activation_26[0][0]
                                                                 batch_normalization_28[0][0]
__________________________________________________________________________________________________
activation_28 (Activation)      (None, 32, 32, 64)   0           add_13[0][0]
__________________________________________________________________________________________________
conv2d_31 (Conv2D)              (None, 32, 32, 64)   36928       activation_28[0][0]
__________________________________________________________________________________________________
batch_normalization_29 (BatchNo (None, 32, 32, 64)   256         conv2d_31[0][0]
__________________________________________________________________________________________________
activation_29 (Activation)      (None, 32, 32, 64)   0           batch_normalization_29[0][0]
__________________________________________________________________________________________________
conv2d_32 (Conv2D)              (None, 32, 32, 64)   36928       activation_29[0][0]
__________________________________________________________________________________________________
batch_normalization_30 (BatchNo (None, 32, 32, 64)   256         conv2d_32[0][0]
__________________________________________________________________________________________________
add_14 (Add)                    (None, 32, 32, 64)   0           activation_28[0][0]
                                                                 batch_normalization_30[0][0]
__________________________________________________________________________________________________
activation_30 (Activation)      (None, 32, 32, 64)   0           add_14[0][0]
__________________________________________________________________________________________________
conv2d_33 (Conv2D)              (None, 32, 32, 64)   36928       activation_30[0][0]
__________________________________________________________________________________________________
batch_normalization_31 (BatchNo (None, 32, 32, 64)   256         conv2d_33[0][0]
__________________________________________________________________________________________________
activation_31 (Activation)      (None, 32, 32, 64)   0           batch_normalization_31[0][0]
__________________________________________________________________________________________________
conv2d_34 (Conv2D)              (None, 32, 32, 64)   36928       activation_31[0][0]
__________________________________________________________________________________________________
batch_normalization_32 (BatchNo (None, 32, 32, 64)   256         conv2d_34[0][0]
__________________________________________________________________________________________________
add_15 (Add)                    (None, 32, 32, 64)   0           activation_30[0][0]
                                                                 batch_normalization_32[0][0]
__________________________________________________________________________________________________
activation_32 (Activation)      (None, 32, 32, 64)   0           add_15[0][0]
__________________________________________________________________________________________________
conv2d_35 (Conv2D)              (None, 32, 32, 64)   36928       activation_32[0][0]
__________________________________________________________________________________________________
batch_normalization_33 (BatchNo (None, 32, 32, 64)   256         conv2d_35[0][0]
__________________________________________________________________________________________________
activation_33 (Activation)      (None, 32, 32, 64)   0           batch_normalization_33[0][0]
__________________________________________________________________________________________________
conv2d_36 (Conv2D)              (None, 32, 32, 64)   36928       activation_33[0][0]
__________________________________________________________________________________________________
batch_normalization_34 (BatchNo (None, 32, 32, 64)   256         conv2d_36[0][0]
__________________________________________________________________________________________________
add_16 (Add)                    (None, 32, 32, 64)   0           activation_32[0][0]
                                                                 batch_normalization_34[0][0]
__________________________________________________________________________________________________
activation_34 (Activation)      (None, 32, 32, 64)   0           add_16[0][0]
__________________________________________________________________________________________________
conv2d_37 (Conv2D)              (None, 32, 32, 64)   36928       activation_34[0][0]
__________________________________________________________________________________________________
batch_normalization_35 (BatchNo (None, 32, 32, 64)   256         conv2d_37[0][0]
__________________________________________________________________________________________________
activation_35 (Activation)      (None, 32, 32, 64)   0           batch_normalization_35[0][0]
__________________________________________________________________________________________________
conv2d_38 (Conv2D)              (None, 32, 32, 64)   36928       activation_35[0][0]
__________________________________________________________________________________________________
batch_normalization_36 (BatchNo (None, 32, 32, 64)   256         conv2d_38[0][0]
__________________________________________________________________________________________________
add_17 (Add)                    (None, 32, 32, 64)   0           activation_34[0][0]
                                                                 batch_normalization_36[0][0]
__________________________________________________________________________________________________
activation_36 (Activation)      (None, 32, 32, 64)   0           add_17[0][0]
__________________________________________________________________________________________________
global_average_pooling2d (Globa (None, 64)           0           activation_36[0][0]
__________________________________________________________________________________________________
flatten (Flatten)               (None, 64)           0           global_average_pooling2d[0][0]
__________________________________________________________________________________________________
dense (Dense)                   (None, 10)           650         flatten[0][0]
==================================================================================================
```



This prompted us to think, is sampling really necessary? Of course, from an engineering point of view, sampling can greatly reduce the size of the feature map, thereby greatly reducing the amount of calculation. However, in this experimental surface, sampling does not help improve the performance of convolution neural network. Max pooling has the effect of suppressing noise, so it is useful , But max pooling can also be implemented without any downsampling, which is just like traditional median filers.
![medianfilter](https://github.com/akkaze/cnn-without-any-downsampling/blob/master/assets/medianfilter.png)

<center>A typical median filtering. Here the size of the convolution kernel is 20. Note that the output image size has not been changed. Max pooing is similar to median filtering. Both can be used to suppress noise.</center>


This also shows that each  convolution layer is used to  encoding spatial correlations, shallow features encode short-range correlations, and deeper convolution layers encode longer-range spatial correlations. At a certain level, there is no longer Spatial correlation in the statistical sense (this depends on the size of meaningful objects in image). At this layer, you can use GAP to aggregate spatial features.

Without the sampling layers, the paradigm of a classification network would look like this,

```
Input-->Conv(dilate_rate=1)-->Conv(dilate_rate=2)-->Conv(dilate_rate=4)-->Conv(dilate_rate=8)-->GAP-->Conv1x1-->Softmax-->Output
```

The paradigm of a semantic segmentation network will look like this,

```
Input-->Conv(dilate_rate=1)-->Conv(dilate_rate=2)-->Conv(dilate_rate=4)-->Conv(dilate_rate=8)-->Conv(dilate_rate=4)-->Conv(dilate_rate=2)-->Conv(dilate_rate=1)-->Softmax-->Output
```
As far as I know, I was the first one to use dilated convolution combined with global avergage pooling for image classification and segmentation. Even if there is no performance improvement (but basically no worsing). Note that dilated convolution is not necessary. A larger kernel size Convolution can replace it, but this will inevitably introduce more parameters, which may lead to overfitting.
Similar ideas first appeared in paper of deeplab, [Rethinking Atrous Convolution for Semantic Image Segmentation]: https://arxiv.org/abs/1706.05587

In this article, dilated convolution is mainly used to extract more compact features by removing the downsampling operation of the last few layers of the network and the upsampling operation of the corresponding filter kernel, without adding new additional learning parameters.

![goingdeeper](https://github.com/akkaze/cnn-without-any-downsampling/blob/master/assets/goingdeeper.png)

<center>The atrous convolution module designed in a serial manner, copies the last block of ResNet, such as block4, and cascades the copied blocks in a serial manner.</center>