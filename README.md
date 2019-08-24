# SFGAN(specific face generative adversarial net)

This repository provides a Tensorflow implementation of SFGAN. Based on StyleGAN,sfgan can generate  any specified face-attribute-image by constraining the latent code.

### Paper

---



### Dependencies

- Python3.6
- tensorflow



### Usage

#### 1. Create environment

```
conda env create -f environment.yaml
```

#### 2. Cloning the repository

```
git clone https://github.com/xuigo/specific_face_gan.git
cd specific_face_gan
```

#### 3. Download file

- download cnn_classifier model from 

  [here]: https://pan.baidu.com/s/19zPQYQ9LTnihfigoac54ow

   and put **model** dir in **attr_cnn/**

- download stylegan model from 

  [here]: https://pan.baidu.com/s/1_rdmh1TltdH42p8TSfdx9A

   and put **model** dir in **ROOT DIR**

#### 4.  Generate images

```
vim test.py

attr='gender'   # choose which attribute to generate in [age,gender,glass] 
save_Path='save_gender'  #where to save Img

python test.py
```

#### 5. Result

##### 		Age

![](C:\Users\Sohey\Pictures\age.png)

##### 		Gender

![](C:\Users\Sohey\Pictures\gender.png)

##### 		Glass

![](C:\Users\Sohey\Pictures\glass.png)