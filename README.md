Building a convolutional neural network (CNN/ConvNet) using TensorFlow NN (tf.nn) module. The CNN model architecture is created and trained using the CIFAR10 dataset. The model is accessed using HTTP by creating a Web application using Python and Flask.
The following diagram summarizes the project.
![system_diagram](https://user-images.githubusercontent.com/16560492/39411182-56ae1492-4c05-11e8-99cd-3172698d97e3.png)

The project steps are as follows:

1) </h4>Preparing the Training Data</h4>
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
```python
def create_conv_layer(input_data, filter_size, num_filters):
    """
    Builds the CNN convolution (conv) layer.
    :param input_data:patch data to be processed.
    :param filter_size:#Number of rows and columns of each filter. It is expected to have a rectangular filter.
    :param num_filters:Number of filters.
    :return:The last fully connected layer of the network.
    """
    """
    Preparing the filters of the conv layer by specifiying its shape. 
    Number of channels in both input image and each filter must match.
    Because number of channels is specified in the shape of the input image as the last value, index of -1 works fine.
    """
    filters = tensorflow.Variable(tensorflow.truncated_normal(shape=(filter_size, filter_size, tensorflow.cast(input_data.shape[-1], dtype=tensorflow.int32), num_filters),
                                                              stddev=0.05))
    print("Size of conv filters bank : ", filters.shape)

    """
    Building the convolution layer by specifying the input data, filters, strides along each of the 4 dimensions, and the padding.
    Padding value of 'VALID' means the some borders of the input image will be lost in the result based on the filter size.
    """
    conv_layer = tensorflow.nn.conv2d(input=input_data,
                                      filter=filters,
                                      strides=[1, 1, 1, 1],
                                      padding="VALID")
    print("Size of conv result : ", conv_layer.shape)
    return filters, conv_layer#Returing the filters and the convolution layer result.

def create_CNN(input_data, num_classes, keep_prop):
    """
    Builds the CNN architecture by stacking conv, relu, pool, dropout, and fully connected layers.
    :param input_data:patch data to be processed.
    :param num_classes:Number of classes in the dataset. It helps determining the number of outputs in the last fully connected layer.
    :param keep_prop:probability of dropping neurons in the dropout layer.
    :return: last fully connected layer.
    """
    #Preparing the first convolution layer.
    filters1, conv_layer1 = create_conv_layer(input_data=input_data, filter_size=5, num_filters=4)
    """
    Applying ReLU activation function over the conv layer output. 
    It returns a new array of the same shape as the input array.
    """
    relu_layer1 = tensorflow.nn.relu(conv_layer1)
    print("Size of relu1 result : ", relu_layer1.shape)
    """
    Max pooling is applied to the ReLU layer result to achieve translation invariance.
    It returns a new array of a different shape from the the input array relative to the strides and kernel size used.
    """
    max_pooling_layer1 = tensorflow.nn.max_pool(value=relu_layer1,
                                                ksize=[1, 2, 2, 1],
                                                strides=[1, 1, 1, 1],
                                                padding="VALID")
    print("Size of maxpool1 result : ", max_pooling_layer1.shape)

    #Similar to the previous conv-relu-pool layers, new layers are just stacked to complete the CNN architecture.
    #Conv layer with 3 filters and each filter is of sisze of 5x5.
    filters2, conv_layer2 = create_conv_layer(input_data=max_pooling_layer1, filter_size=7, num_filters=3)
    relu_layer2 = tensorflow.nn.relu(conv_layer2)
    print("Size of relu2 result : ", relu_layer2.shape)
    max_pooling_layer2 = tensorflow.nn.max_pool(value=relu_layer2,
                                                ksize=[1, 2, 2, 1],
                                                strides=[1, 1, 1, 1],
                                                padding="VALID")
    print("Size of maxpool2 result : ", max_pooling_layer2.shape)

    #Conv layer with 2 filters and a filter sisze of 5x5.
    filters3, conv_layer3 = create_conv_layer(input_data=max_pooling_layer2, filter_size=5, num_filters=2)
    relu_layer3 = tensorflow.nn.relu(conv_layer3)
    print("Size of relu3 result : ", relu_layer3.shape)
    max_pooling_layer3 = tensorflow.nn.max_pool(value=relu_layer3,
                                                ksize=[1, 2, 2, 1],
                                                strides=[1, 1, 1, 1],
                                                padding="VALID")
    print("Size of maxpool3 result : ", max_pooling_layer3.shape)

    #Adding dropout layer before the fully connected layers to avoid overfitting.
    flattened_layer = dropout_flatten_layer(previous_layer=max_pooling_layer3, keep_prop=keep_prop)

    #First fully connected (FC) layer. It accepts the result of the dropout layer after being flattened (1D).
    fc_resultl = fc_layer(flattened_layer=flattened_layer, num_inputs=flattened_layer.get_shape()[1:].num_elements(),
                          num_outputs=200)
    #Second fully connected layer accepting the output of the previous fully connected layer. Number of outputs is equal to the number of dataset classes.
    fc_result2 = fc_layer(flattened_layer=fc_resultl, num_inputs=fc_resultl.get_shape()[1:].num_elements(),
                          num_outputs=num_classes)
    print("Fully connected layer results : ", fc_result2)
    return fc_result2#Returning the result of the last FC layer.

def dropout_flatten_layer(previous_layer, keep_prop):
    """
    Applying the dropout layer.
    :param previous_layer: Result of the previous layer to the dropout layer.
    :param keep_prop: Probability of keeping neurons.
    :return: flattened array.
    """
    dropout = tensorflow.nn.dropout(x=previous_layer, keep_prob=keep_prop)
    num_features = dropout.get_shape()[1:].num_elements()
    layer = tensorflow.reshape(previous_layer, shape=(-1, num_features))#Flattening the results.
    return layer

def fc_layer(flattened_layer, num_inputs, num_outputs):
    """
    uilds a fully connected (FC) layer.
    :param flattened_layer: Previous layer after being flattened.
    :param num_inputs: Number of inputs in the previous layer.
    :param num_outputs: Number of outputs to be returned in such FC layer.
    :return:
    """
    #Preparing the set of weights for the FC layer. It depends on the number of inputs and number of outputs.
    fc_weights = tensorflow.Variable(tensorflow.truncated_normal(shape=(num_inputs, num_outputs),
                                                              stddev=0.05))
    #Matrix multiplication between the flattened array and the set of weights.
    fc_resultl = tensorflow.matmul(flattened_layer, fc_weights)
    return fc_resultl#Output of the FC layer (result of matrix multiplication).
```

3) <h4>Training the CNN</h4>
Training the CNN based on the prepared training data.
```python
num_patches = 1#Number of patches
for patch_num in numpy.arange(num_patches):
    print("Patch : ", str(patch_num))
    percent = 20 #percent of samples to be included in each path.
    #Getting the input-output data of the current path.
    shuffled_data, shuffled_labels = get_patch(data=dataset_array, labels=dataset_labels, percent=percent)
    #Data required for cnn operation. 1)Input Images, 2)Output Labels, and 3)Dropout probability
    cnn_feed_dict = {data_tensor: shuffled_data,
                     label_tensor: shuffled_labels,
                     keep_prop: 0.5}
    """
    Training the CNN based on the current patch. 
    CNN error is used as input in the run to minimize it.
    SoftMax predictions are returned to compute the classification accuracy.
    """
    softmax_predictions_, _ = sess.run([softmax_predictions, error], feed_dict=cnn_feed_dict)
    #Calculating number of correctly classified samples.
    correct = numpy.array(numpy.where(softmax_predictions_ == shuffled_labels))
    correct = correct.size
    print("Correct predictions/", str(percent * 50000/100), ' : ', correct)
```

4) <h4>Saving the Trained CNN Model</h4>
The trained CNN model is saved for later use for predicting unseen samples.
```python
#Saving the model after being trained.
saver = tensorflow.train.Saver()
save_model_path = "C:\\Users\\Dell\\Desktop\\model\\"
save_path = saver.save(sess=sess, save_path=save_model_path+"model.ckpt")
print("Model saved in : ", save_path)
```

5) <h4>Restoring the Pre-Trained Model</h4>
Before predicting class label for unseen samples, the saved CNN model must be restored.
```python
#Restoring the previously saved trained model.
saved_model_path = 'C:\\Users\\Dell\\Desktop\\model\\'
saver = tensorflow.train.import_meta_graph(saved_model_path+'model.ckpt.meta')
saver.restore(sess=sess, save_path=saved_model_path+'model.ckpt')
```

6) <h4>Testing the Trained CNN Model</h4>
New unseen test samples are fed to the model for predicting its labels.
```python
softmax_propabilities = graph.get_tensor_by_name(name="softmax_probs:0")
softmax_predictions = tensorflow.argmax(softmax_propabilities, axis=1)
data_tensor = graph.get_tensor_by_name(name="data_tensor:0")
label_tensor = graph.get_tensor_by_name(name="label_tensor:0")
keep_prop = graph.get_tensor_by_name(name="keep_prop:0")

#keep_prop is equal to 1 because there is no more interest to remove neurons in the testing phase.
feed_dict_testing = {data_tensor: dataset_array,
                     label_tensor: dataset_labels,
                     keep_prop: 1.0}
#Running the session to predict the outcomes of the testing samples.
softmax_propabilities_, softmax_predictions_ = sess.run([softmax_propabilities, softmax_predictions],
                                                      feed_dict=feed_dict_testing)
#Assessing the model accuracy by counting number of correctly classified samples.
correct = numpy.array(numpy.where(softmax_predictions_ == dataset_labels))
correct = correct.size
print("Correct predictions/10,000 : ", correct)
```

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
```python
def CNN_predict():
    """
    Reads the uploaded image file and predicts its label using the saved pre-trained CNN model.
    :return: Either an error if the image is not for CIFAR10 dataset or redirects the browser to a new page to show the prediction result if no error occurred.
    """
    """
    Setting the previously created 'secure_filename' to global.
    This is because to be able invoke a global variable created in another function, it must be defined global in the caller function.
    """
    global secure_filename
    #Reading the image file from the path it was saved in previously.
    img = scipy.misc.imread(os.path.join(app.root_path, secure_filename))

    """
    Checking whether the image dimensions match the CIFAR10 specifications.
    CIFAR10 images are RGB (i.e. they have 3 dimensions). It number of dimenions was not equal to 3, then a message will be returned.
    """
    if(img.ndim) == 3:
        """
        Checking if the number of rows and columns of the read image matched CIFAR10 (32 rows and 32 columns).
        """
        if img.shape[0] == img.shape[1] and img.shape[0] == 32:
            """
            Checking whether the last dimension of the image has just 3 channels (Red, Green, and Blue).
            """
            if img.shape[-1] == 3:
                """
                Passing all conditions above, the image is proved to be of CIFAR10.
                This is why it is passed to the predictor.
                """
                predicted_class = CIFAR10_CNN_Predict_Image.main(img)
                """
                After predicting the class label of the input image, the prediction label is rendered on an HTML page.
                The HTML page is fetched from the /templates directory. The HTML page accepts an input which is the predicted class.
                """
                return flask.render_template(template_name_or_list="prediction_result.html", predicted_class=predicted_class)
            else:
                # If the image dimensions do not match the CIFAR10 specifications, then an HTML page is rendered to show the problem.
                return flask.render_template(template_name_or_list="error.html", img_shape=img.shape)
        else:
            # If the image dimensions do not match the CIFAR10 specifications, then an HTML page is rendered to show the problem.
            return flask.render_template(template_name_or_list="error.html", img_shape=img.shape)
    return "An error occurred."#Returned if there is a different error other than wrong image dimensions.
"""
Creating a route between the URL (http://localhost:7777/predict) to a viewer function that is called after navigating to such URL. 
Endpoint 'predict' is used to make the route reusable without hard-coding it later.
"""
app.add_url_rule(rule="/predict/", endpoint="predict", view_func=CNN_predict)
```

*****************************************<br>
<h4>Updates 2-5-2018</h4>

The previous implementation can only be used in development but not in production because server was opening a session for each new request. This wastes the resources to much.</br>
The modified  code can be used in production mode because the session is opened globally only once to serve all requests. This is efficient than before.<br>
The way it is done is opening the session while opening the server. Exactly it is done using such code:

```python
if __name__ == "__main__":
    #Restoring the previously saved trained model.
    prepare_TF_session(saved_model_path='C:\\Users\\Dell\\Desktop\\model\\')
    app.run(host="localhost", port=7777, debug=True)
```

*****************************************

<h3>References</h3>
tf.nn module:<br>
https://www.tensorflow.org/api_docs/python/tf/nn<br>
CIFAR10 dataset:<br>
https://www.cs.toronto.edu/~kriz/cifar.html<br>

<h3>For more info.</h3>
KDnuggets: https://www.kdnuggets.com/author/ahmed-gad<br>LinkedIn: https://www.linkedin.com/in/ahmedfgad<br>Facebook: https://www.facebook.com/ahmed.f.gadd<br>ahmed.f.gad@gmail.com<br>ahmed.fawzy@ci.menofia.edu.eg
