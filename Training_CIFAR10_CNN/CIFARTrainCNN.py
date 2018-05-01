import pickle
import tensorflow
import numpy
import matplotlib.pyplot
import scipy.misc
import os

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

def get_patch(data, labels, percent=70):
    """
    Returning patch to train the CNN.
    :param data: Complete input data after being encoded and reshaped.
    :param labels: Labels of the entire dataset.
    :param percent: Percent of samples to get returned in each patch.
    :return: Subset of the data (patch) to train the CNN model.
    """
    #Using the percent of samples per patch to return the actual number of samples to get returned.
    num_elements = numpy.uint32(percent*data.shape[0]/100)
    shuffled_labels = labels#Temporary variable to hold the data after being shuffled.
    numpy.random.shuffle(shuffled_labels)#Randomly reordering the labels.
    """
    The previously specified percent of the data is returned starting from the beginning until meeting the required number of samples. 
    The labels indices are also used to return their corresponding input images samples.
    """
    return data[shuffled_labels[:num_elements], :, :, :], shuffled_labels[:num_elements]

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

#***********************************************************************************
#Nnumber of classes in the dataset. Used to specify number of outputs in the last fully connected layer.
num_datatset_classes = 10
#Number of rows & columns in each input image. The image is expected to be rectangular Used to reshape the images and specify the input tensor shape.
im_dim = 32
#Number of channels in rach input image. Used to reshape the images and specify the input tensor shape.
num_channels = 3

#Directory at which the training binary files of the CIFAR10 dataset are saved.
patches_dir = "C:\\Users\\Dell\\Downloads\\Compressed\\cifar-10-python\\cifar-10-batches-py\\"
#Reading the CIFAR10 training binary files and returning the input data and output labels. Output labels are used to test the CNN prediction accuracy.
dataset_array, dataset_labels = get_dataset_images(dataset_path=patches_dir, im_dim=im_dim, num_channels=num_channels)
print("Size of data : ", dataset_array.shape)

"""
Input tensor to hold the data read above. It is the entry point of the computational graph.
The given name of 'data_tensor' is useful for retreiving it when restoring the trained model graph for testing.
"""
data_tensor = tensorflow.placeholder(tensorflow.float32, shape=[None, im_dim, im_dim, num_channels], name='data_tensor')
"""
Tensor to hold the outputs label. 
The name "label_tensor" is used for accessing the tensor when tesing the saved trained model after being restored.
"""
label_tensor = tensorflow.placeholder(tensorflow.float32, shape=[None], name='label_tensor')

#The probability of dropping neurons in the dropout layer. It is given a name for accessing it later.
keep_prop = tensorflow.Variable(initial_value=0.5, name="keep_prop")

#Building the CNN architecure and returning the last layer which is the fully connected layer.
fc_result2 = create_CNN(input_data=data_tensor, num_classes=num_datatset_classes, keep_prop=keep_prop)

"""
Predicitions probabilities of the CNN for each training sample.
Each sample has a probability for each of the 10 classes in the dataset.
Such tensor is given a name for accessing it later.
"""
softmax_propabilities = tensorflow.nn.softmax(fc_result2, name="softmax_probs")

"""
Predicitions labels of the CNN for each training sample.
The input sample is classified as the class of the highest probability.
axis=1 indicates that maximum of values in the second axis is to be returned. This returns that maximum class probability fo each sample.
"""
softmax_predictions = tensorflow.argmax(softmax_propabilities, axis=1)

#Cross entropy of the CNN based on its calculated probabilities.
cross_entropy = tensorflow.nn.softmax_cross_entropy_with_logits(logits=tensorflow.reduce_max(input_tensor=softmax_propabilities, reduction_indices=[1]),
                                                                labels=label_tensor)
#Summarizing the cross entropy into a single value (cost) to be minimized by the learning algorithm.
cost = tensorflow.reduce_mean(cross_entropy)
#Minimizng the network cost using the Gradient Descent optimizer with a learning rate is 0.01.
error = tensorflow.train.GradientDescentOptimizer(learning_rate=.01).minimize(cost)

#Creating a new TensorFlow Session to process the computational graph.
sess = tensorflow.Session()
#Wiriting summary of the graph to visualize it using TensorBoard.
tensorflow.summary.FileWriter(logdir="./log/", graph=sess.graph)
#Initializing the variables of the graph.
sess.run(tensorflow.global_variables_initializer())

"""
Because it may be impossible to feed the complete data to the CNN on normal machines, it is recommended to split the data into a number of patches.
A percent of traning samples is used to create each path. Samples for each path can be randomly selected.
"""
num_patches = 10#Number of patches
for patch_num in numpy.arange(num_patches):
    print("Patch : ", str(patch_num))
    percent = 80 #percent of samples to be included in each path.
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

#Saving the model after being trained.
saver = tensorflow.train.Saver()
save_model_path = "C:\\Users\\Dell\\Desktop\\model\\"
save_path = saver.save(sess=sess, save_path=save_model_path+"model.ckpt")
print("Model saved in : ", save_path)

