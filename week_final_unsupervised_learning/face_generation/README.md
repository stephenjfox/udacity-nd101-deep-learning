# Face Generation (for Udacity's nd101: Deep Learning Foundations)

This project is to demonstration the creation of __one__ DCGAN architecture to
learn from __two__ distinct data sets.

1. The MNIST data set of handwritten digits
2. The CelebA data set of celebrity faces


Using iPython Notebooks (powered by the Jupyter platform), we will demonstrate the
creation of novel images: both handwriting and human-realistic faces


## Setup

With [FloydHub](https://www.floydhub.com), initialize the project with the pre-uploaded
data set provide by Udacity

`floyd run --gpu --env tensorflow --mode jupyter --data floydhub/datasets/udacity-gan/1:/input`

In a few moments, your project will spin up with access to the correct data
