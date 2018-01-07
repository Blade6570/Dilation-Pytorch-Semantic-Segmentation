# Dialation-Pytorch
A PyTorch implementation of semantic segmentation according to [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122) by Yu and Koltun.
Pretrained weights are obtained from [dialation-Tensorflow](https://github.com/ndrplz/dilation-tensorflow) and thanks to [Andrea Palazzi](https://github.com/ndrplz) for his work from which this work has been inspired.

![Input](https://github.com/Blade6570/Dialation-Pytorch/blob/master/data/cityscapes_real.png?raw=true "Input Image")
**How To**
1. Download the pretrained weights from [cityscapes](https://drive.google.com/file/d/0Bx9YaGcDPu3XR0d4cXVSWmtVdEE/view)
2. Keep that in the *data* folder
3. Run *CAN_Network.py* for the demo. 

**Notes**
I have kept the output image size 1024x1024x3 due to memory constraint. If you want the size to be 1024x2048x3, have enough memory and change the sizes appropriately in the code. 
