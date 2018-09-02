# 

# Semantic Segmentation Project



## Introduction

In this project, The model will label the pixels of a road in images using a Fully Convolutional Network (FCN) and a pre-trained VGG16 model.

### Dataset

[Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) which can be downloaded from [here](http://www.cvlibs.net/download.php?file=data_road.zip) 

### Architecture

The model is a Fully Convolutional Network  (You can Check this paper for more info  [Link](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)) 

which is built above a pre-trained VGG 16 model, by removing the last layer and converting it by a 1x1 convolution with 2 Classes as the depth (Road, Not Road). Then Using Upsample to restore the spatial dimensions of the input image. Some skip connections between VGG layers and the new Layers were used to improve the Performance.

### Training

The hyperparameters used for training are:

- keep_prob: 0.5
- learning_rate: 0.001
- epochs: 60
- batch_size: 5

The model was Trained using Google Colab GPU Runtime. it took about 1-2 hours of Training.

### Results

After the 60 epochs the model reached 1.4 as epoch loss. 

### Samples

Below are a few sample images from the output , with the segmentation class overlaid upon the original image in violet.

![um_000019](samples/um_000019.png)

![uu_000079](samples/uu_000079.png)

![um_000017](samples/um_000017.png)

![umm_000010](samples/umm_000010.png)

![umm_000082](samples/umm_000082.png)

![umm_000049](samples/umm_000049.png)

![uu_000089](samples/uu_000089.png)

![um_000018](samples/um_000018.png)

![uu_000026](samples/uu_000026.png)

![umm_000052](samples/umm_000052.png)

![uu_000065](samples/uu_000065.png)




### Introduction

In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup

##### GPU

`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.

##### Frameworks and Packages

Make sure you have the following is installed:

- [Python 3](https://www.python.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)

  ##### Dataset

  Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start

##### Implement

Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.

##### Run

Run the following command to run the project:

```
python main.py
```

**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission

1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
   - `helper.py`
   - `main.py`
   - `project_tests.py`
   - Newest inference images from `runs` folder  (**all images from the most recent run**)

   ### Tips
4. The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip).
5. The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [post](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/forum_archive/Semantic_Segmentation_advice.pdf) for more information.  A summary of additional points, follow. 
6. The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
7. When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.

### Using GitHub and Creating Effective READMEs

If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
