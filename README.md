Building a convolutional neural network (CNN/ConvNet) using TensorFlow NN (tf.nn) module. The CNN model architecture is created and trained using the CIFAR10 dataset. The model is accessed using HTTP by creating a Web application using Python and Flask.
The following diagram summarizes the project.
![system_diagram](https://user-images.githubusercontent.com/16560492/39411182-56ae1492-4c05-11e8-99cd-3172698d97e3.png)

The project steps are as follows:

1) Preparing the Training Data
The training data should be read and prepared for use with the CNN.

2) Building the Computational Graph
The CNN architecture is created by stacking conv-relu-pool-dropout-fc layers.

3) Training the CNN
Training the CNN based on the prepared training data.

4) Saving the Trained CNN Model
The trained CNN model is saved for later use for predicting unseen samples.

5) Restoring the Pre-Trained Model
Before predicting class label for unseen samples, the saved CNN model must be restored.

6) Testing the Trained CNN Model
New unseen test samples are fed to the model for predicting its labels.

7) Building Flask Web Application
A Flask Web application is created to enable the remote access of the trained CNN model for classifying images transferred using the HTTP protocol.

8) Upload an Image via HTML Form
A HTML page will allow the user to upload a CIFAR10 image to the server. 

9) Using JavaScript and CSS
Some helper JS and CSS files are created to style the Web application.

10) Invoking the Trained Model for Prediction
The uploaded image will be classified using the restored pre-trained CNN model. The classification label will finally get rendered on a new HTML page.
