## How to  use this code

This repo requires **tensorflow-gpu-1.5.0** or other compatible version of tensorflow,

you can run it simple by type 

**python cifar10.py**

### Does CNN really need downsampling (upsampling)?

In common convolutional neural networks, sampling is almost ubiquitous, formerly max_pooling, and now strided convolution.
Take the vgg network as an example, which uses quite a lot of max_pooling,

![vgg](assets\vgg.png)
<center>The input side is below, you can see that a lot of 2x2 pooling is used in the network</center>
Also, when doing semantic segmentation or object detection, we use quite a lot of upsampling, or transposed convolution.

![fcn](assets\fcn.png)
<center>Typical fcn structure, pay attention to the decovolution distinguished by red</center>
Previously, we used fc in the last few layers of the classification network. Later, fc was proved to have too many parameters and poor generalization performance. It was replaced by global average pooling and it first appeared in network in network.

![gap](assets\gap.jpg)
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

![acc](assets\acc.png)

The final accuracy rate reached 76%. The accuracy rate of a convolutional network with vgg structure with the same parameters is basically around this.

This prompted us to think, is sampling really necessary? Of course, from an engineering point of view, sampling can greatly reduce the size of the feature map, thereby greatly reducing the amount of calculation. However, in this experimental surface, sampling does not help improve the performance of convolution neural network. Max pooling has the effect of suppressing noise, so it is useful , But max pooling can also be implemented without any downsampling, which is just like traditional median filers.

This also shows that each  convolution layer is used to  encoding spatial correlations, shallow features encode short-range correlations, and deeper convolution layers encode longer-range spatial correlations. At a certain level, there is no longer Spatial correlation in the statistical sense (this depends on the size of meaningful objects in image). At this layer, you can use GAP to aggregate spatial features.

Without the sampling layers, the paradigm of a classification network would look like this,

```
Input-->Conv(dilate_rate=1)-->Conv(dilate_rate=2)-->Conv(dilate_rate=4)-->Conv(dilate_rate=8)-->GAP-->Conv1x1-->Softmax-->Output
```

The paradigm of a semantic segmentation network will look like this,

```
Input-->Conv(dilate_rate=1)-->Conv(dilate_rate=2)-->Conv(dilate_rate=4)-->Conv(dilate_rate=8)-->Conv(dilate_rate=4)-->Conv(dilate_rate=2)-->Conv(dilate_rate=1)-->Softmax-->Output
```

