[Semantic Segmentation Experiments on VOC2007 using CNNs and Transfer Learning] - CSE251B PA3
by Eunji Song, Zachary Adler  
  
Abstract:  
We examine the task of semantic segmentation on VOC2007, which is classifying each pixel into 1 of 21 classes. We started with a baseline encoder/decoder model(Pixel Accuracy 0.6937, IoU 0.0641) and then experimented with improvements of learning rate scheduling(Pixel Accuracy 0.7262, IoU 0.0540), data augmentation(Pixel Accuracy 0.7319, IoU 0.0774), and class-weighted loss(Pixel Accuracy 0.6748, IoU 0.0609). To further improve the model, we designed two different custom architecture. First one is inception inspired model(Pixel Accuracy 0.6937, IoU 0.0847) that uses features extracted by different sizes of kernels. Second one is using max location switches(Pixel Accuracy 0.6962, IoU 0.0860) to remember the max pooling location. We then explored architecture remodeling with transfer learning from ResNet34 (Pixel Accuracy 0.7227, IoU 0.1305), implementing the UNet architecture(Pixel Accuracy 0.6849, IoU 0.0638). We found that UNet segmented the images well but did not classify the pixels well, while transfer from ResNet34 improved the classification accuracy. The best architecture we found was with transfer learning from ResNet34. These results can be further improved if we use the data augmentation that we omitted due to resource limits.

How to Run:  

Run download.py once to obtain the data.
Our code runs all the experiments on the report by setting the MODE values in the train_combined.py code line 19.
To reproduce our results on each section, below is the setting.
3. Baseline: MODE = []
4.a Learning Rate Schedule: MODE = ['lr']
4.b Data Augmentation: MODE = ['augment']
4.c Balanced Loss Criterion: MODE = ['weight']
5.a1 Custom1: MODE = ['lr', 'weight', 'augment', 'custom1']
5.a2 Custom2: MODE = ['lr', 'weight', 'augment', 'custom2']
5.b Transfer: MODE = ['lr', 'weight', 'transfer']
5.c U-Net: MODE = ['lr', 'weight', 'unet']

Code Run Example:  
!python3 download.py
!python3 train_combined.py

(train_combined.py:19) MODE = ['lr', 'weight', 'augment', 'custom1']
