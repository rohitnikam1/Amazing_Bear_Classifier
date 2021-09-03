# Amazing Bear Classifier using Convolutional Neural Network

In this project, we build a bear detector using fastai library for deep learning. It will discriminate between three types of bear: grizzly, black, and teddy bears. 

Additional to conventional data preprocessing steps, we implement GPU accelerated data augmentation transforms (unique feature of fastai library) to images. We then fine-tune a pre-trained resnet18 model to classify the three bear categories. The dataset contains 900 images. Our system reaches 96.25% accuracy in 4 epochs. The analysis can be seen at `bear_classification_dl.ipynb`.

We later attempt to export the classification model into a working online application using IPython widgets (ipywidgets), and Voilà.
