# SRCNN-Chainer
SRCNN by Chainer

SRCNN by Chainer(v3) and python3.  
A Chainer implementation of [SRCNN](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html).  
Please notice that i have used padding in the model, which is different from the original paper.  
If you have any question, please feel free to contact me.

## Usage

### compare   
```
python compare_image_srcnn --input_file/-i  filename
```

It will downsize the given image to the low resolution image with a factor=2, then upsize it by bicubic and SRCNN respectively to generate super resolution image with a factor=2 and compare the PSNR/SSIM between the SR image and ground truth.

### generate  
```
python generate_2x_srcnn --input_file/-i  filename
```
It will generate a 2x SR image of the given image by SRCNN. 

## Result

### compare
#### ground truth
![image](https://github.com/irasin/SRCNN-Chainer/blob/master/result/butterfly.png)

#### low resolution
![image](https://github.com/irasin/SRCNN-Chainer/blob/master/result/butterfly_low.png)

#### bicubic
![image](https://github.com/irasin/SRCNN-Chainer/blob/master/result/butterfly_bic.png)

#### super resolution
![image](https://github.com/irasin/SRCNN-Chainer/blob/master/result/butterfly_super.png)

#### generate 2x
![image](https://github.com/irasin/SRCNN-Chainer/blob/master/result/butterfly_super_2x.png)


