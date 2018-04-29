import tensorflow
import pickle
import numpy

def unpickle_patch(file):
    """
    Identical to the one used in training.
    """
    patch_bin_file = open(file, 'rb')
    patch_dict = pickle.load(patch_bin_file, encoding='bytes')
    return patch_dict

def main(img):
    """
    The 'main' method accepts an input image array of size 32x32x3 and returns its class label.
    :param img:RGB image of size 32x32x3.
    :return:Predicted class label.
    """
    #Dataset path containing a binary file with the labels of classes. Useful to decode the prediction code into a significant textual label.
    patches_dir = "C:\\Users\\Dell\\Downloads\\Compressed\\cifar-10-python\\cifar-10-batches-py\\"
    dataset_array = numpy.random.rand(1, 32, 32, 3)
    dataset_array[0, :, :, :] = img

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
                         keep_prop: 1.0}
    #Running the session to predict the outcomes of the testing samples.
    softmax_propabilities_, softmax_predictions_ = sess.run([softmax_propabilities, softmax_predictions],
                                                          feed_dict=feed_dict_testing)
    label_names_dict = unpickle_patch(patches_dir + "batches.meta")
    dataset_label_names = label_names_dict[b"label_names"]
    return dataset_label_names[softmax_predictions_[0]].decode('utf-8')
