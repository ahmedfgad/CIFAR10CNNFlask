import tensorflow
import pickle
import numpy

def get_dataset_images(test_path_path, im_dim=32, num_channels=3):
    """
    Similar to the one used in training except that there is just a single testing binary file for testing the CIFAR10 trained models.
    """
    print("Working on testing patch")
    data_dict = unpickle_patch(test_path_path)
    images_data = data_dict[b"data"]
    dataset_array = numpy.reshape(images_data, newshape=(len(images_data), im_dim, im_dim, num_channels))
    return dataset_array, data_dict[b"labels"]

def unpickle_patch(file):
    """
    Identical to the one used in training.
    """
    patch_bin_file = open(file, 'rb')
    patch_dict = pickle.load(patch_bin_file, encoding='bytes')
    return patch_dict

#***********************************************************************************
#Dataset path containing the testing binary file to be decoded.
patches_dir = "C:\\Users\\Dell\\Downloads\\Compressed\\cifar-10-python\\cifar-10-batches-py\\"
dataset_array, dataset_labels = get_dataset_images(test_path_path=patches_dir + "test_batch", im_dim=32, num_channels=3)
print("Size of data : ", dataset_array.shape)

sess = tensorflow.Session()

#Restoring the previously saved trained model.
saved_model_path = 'C:\\Users\\Dell\\Desktop\\model\\'
saver = tensorflow.train.import_meta_graph(saved_model_path+'model.ckpt.meta')
saver.restore(sess=sess, save_path=saved_model_path+'model.ckpt')

#Initalizing the varaibales.
sess.run(tensorflow.global_variables_initializer())

graph = tensorflow.get_default_graph()

"""
Restoring previous created tensors in the training phase based on their given tensor names in the training phase.
Some of such tensors will be assigned the testing input data and their outcomes (data_tensor, label_tensor, and keep_prop).
Others are helpful in assessing the model prediction accuracy (softmax_propabilities and softmax_predictions).
"""
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

#Closing the session
sess.close()
