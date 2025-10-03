### This Repository is Style GAN in PyTorch , From this paper : [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948) 
### Data : Caleba Dataset (with 10k subset) 

### in epoch 10 / 300 
![image alt](https://github.com/HajarHAMDOUCH01/Face-Generator-StyleGAN-PyTorch/blob/b68a476369d4d2e698b880b498a803eb3028515d/samples_epoch_10%20(1).png)

### in epoch 15 / 300 
![image alt](https://github.com/HajarHAMDOUCH01/Face-Generator-StyleGAN-PyTorch/blob/b68a476369d4d2e698b880b498a803eb3028515d/samples_epoch_15%20(3).png)


To decrease training time and prove model functionning , I :
- Image size : 128*128
- Subset of 10k images from CalebA dataset
- Because the subset was smaller , 8 layers of fully connected layers of mapping net will make the model memorize a space W of the data too well , like using too much effort while 10k is not worth that much effort. so I changed mapping net depth from 8 (as the paper) to 4 because i saw mode collapse in epoch 40 / 300.

### Inference :
- Run testing/test.py , it geneartes a new face using scripted model using Torchscript.

### Model weights :
- i will upload best .pth files in a checkpoint in hugging face for fine tunning or inference. 

### stay tuned for a web app for face generation.

