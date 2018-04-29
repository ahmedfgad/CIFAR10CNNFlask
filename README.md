Building a convolutional neural network (CNN/ConvNet) using TensorFlow NN (tf.nn) module. The CNN model architecture is created and trained using the CIFAR10 dataset. The model is accessed using HTTP by creating a Web application using Python and Flask.
The following diagram summarizes the project.
![system_diagram](https://user-images.githubusercontent.com/16560492/39411182-56ae1492-4c05-11e8-99cd-3172698d97e3.png)

The project steps are as follows:

1) <h4>Preparing the Training Data</h4>
The training data should be read and prepared for use with the CNN.
```python
def get_dataset_images(dataset_path, im_dim=32, num_channels=3):
    """
    This function accepts the dataset path, reads the data, and returns it after being reshaped to match the requierments of the CNN.
    :param dataset_path:Path of the CIFAR10 dataset binary files.
    :param im_dim:Number of rows and columns in each image. The image is expected to be rectangular.
    :param num_channels:Number of color channels in the image.
    :return:Returns the input data after being reshaped and output labels.
    """
    num_files = 5#Number of training binary files in the CIFAR10 dataset.
    images_per_file = 10000#Number of samples withing each binary file.
    files_names = os.listdir(patches_dir)#Listing the binary files in the dataset path.
    """
    Creating an empty array to hold the entire training data after being reshaped.
    The dataset has 5 binary files holding the data. Each binary file has 10,000 samples. Total number of samples in the dataset is 5*10,000=50,000.
    Each sample has a total of 3,072 pixels. These pixels are reshaped to form a RGB image of shape 32x32x3.
    Finally, the entire dataset has 50,000 samples and each sample of shape 32x32x3 (50,000x32x32x3).
    """
    dataset_array = numpy.zeros(shape=(num_files * images_per_file, im_dim, im_dim, num_channels))
    #Creating an empty array to hold the labels of each input sample. Its size is 50,000 to hold the label of each sample in the dataset.
    dataset_labels = numpy.zeros(shape=(num_files * images_per_file), dtype=numpy.uint8)
    index = 0#Index variable to count number of training binary files being processed.
    for file_name in files_names:
        """
        Because the CIFAR10 directory does not only contain the desired training files and has some  other files, it is required to filter the required files.
        Training files start by 'data_batch_' which is used to test whether the file is for training or not.
        """
        if file_name[0:len(file_name) - 1] == "data_batch_":
            print("Working on : ", file_name)
            """
            Appending the path of the binary files to the name of the current file.
            Then the complete path of the binary file is used to decoded the file and return the actual pixels values.
            """
            data_dict = unpickle_patch(dataset_path+file_name)
            """
            Returning the data using its key 'data' in the dictionary.
            Character b is used before the key to tell it is binary string.
            """
            images_data = data_dict[b"data"]
            #Reshaping all samples in the current binary file to be of 32x32x3 shape.
            images_data_reshaped = numpy.reshape(images_data, newshape=(len(images_data), im_dim, im_dim, num_channels))
            #Appending the data of the current file after being reshaped.
            dataset_array[index * images_per_file:(index + 1) * images_per_file, :, :, :] = images_data_reshaped
            #Appening the labels of the current file.
            dataset_labels[index * images_per_file:(index + 1) * images_per_file] = data_dict[b"labels"]
            index = index + 1#Incrementing the counter of the processed training files by 1 to accept new file.
    return dataset_array, dataset_labels#Returning the training input data and output labels.

def unpickle_patch(file):
    """
    Decoding the binary file.
    :param file:File to decode it data.
    :return: Dictionary of the file holding details including input data and output labels.
    """
    patch_bin_file = open(file, 'rb')#Reading the binary file.
    patch_dict = pickle.load(patch_bin_file, encoding='bytes')#Loading the details of the binary file into a dictionary.
    return patch_dict#Returning the dictionary.
```

2) <h4>Building the Computational Graph</h4>
The CNN architecture is created by stacking conv-relu-pool-dropout-fc layers.<br>
![graph_large_attrs_key _too_large_attrs limit_attr_size 1024 run 1 - copy](https://user-images.githubusercontent.com/16560492/39411206-ae3add94-4c05-11e8-9444-a7c21d3fa254.png)

3) <h4>Training the CNN</h4>
Training the CNN based on the prepared training data.

4) <h4>Saving the Trained CNN Model</h4>
The trained CNN model is saved for later use for predicting unseen samples.

5) <h4>Restoring the Pre-Trained Model</h4>
Before predicting class label for unseen samples, the saved CNN model must be restored.

6) <h4>Testing the Trained CNN Model</h4>
New unseen test samples are fed to the model for predicting its labels.

7) <h4>Building Flask Web Application</h4>
A Flask Web application is created to enable the remote access of the trained CNN model for classifying images transferred using the HTTP protocol.

8) <h4>Upload an Image via HTML Form</h4>
A HTML page will allow the user to upload a CIFAR10 image to the server.<br>
![2018-04-29_22-28-43](https://user-images.githubusercontent.com/16560492/39411196-8b5ea3f0-4c05-11e8-8eae-f9006f8f9b63.png)

9) <h4>Using JavaScript and CSS</h4>
Some helper JS and CSS files are created to style the Web application.

10) <h4>Invoking the Trained Model for Prediction</h4>
The uploaded image will be classified using the restored pre-trained CNN model. The classification label will finally get rendered on a new HTML page.<br>
![2018-04-29_22-30-57](https://user-images.githubusercontent.com/16560492/39411202-98faaedc-4c05-11e8-9f3b-785a06bec1cb.png)

<h3>References</h3>
tf.nn module:<br>
https://www.tensorflow.org/api_docs/python/tf/nn<br>
CIFAR10 dataset:<br>
https://www.cs.toronto.edu/~kriz/cifar.html<br>

<h3>For more info.</h3>
KDnuggets: https://www.kdnuggets.com/author/ahmed-gad<br>LinkedIn: https://www.linkedin.com/in/ahmedfgad<br>Facebook: https://www.facebook.com/ahmed.f.gadd<br>ahmed.f.gad@gmail.com<br>ahmed.fawzy@ci.menofia.edu.eg
