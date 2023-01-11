from fileinput import filename
from scipy.io.matlab.mio import loadmat
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import vgg_19
import numpy as np
import cv2
import tables
import scipy.io
import scipy.io as spio
import scipy.io as si
import h5py
import glob
import os
import time
from tensorflow.python import pywrap_tensorflow
import inspect
def Load_Dataset():
   # p = h5py.File('lfw_att_73.mat')
    address = []
    address_train=[]
    address_test=[]
    label_train=[]
    label_test=[]
    attribute_test=[]
    attribute_train=[]
    f = h5py.File('DATA.mat')
    for column in f['name']:
       row_data = []
       for row_number in range(len(column)):
           row_data.append(""+''.join(map(unichr, f[column[row_number]][:]))+"")
           address.append(row_data[0])
           row_data = []

    label = f['labels']
    attribute=f['attribute']
    Label=np.transpose(label)
    Attribute = np.transpose(attribute)
    count = 0
    i = 0
    train_sample=0
    set_variable = True
    label = f['labels']
    Label = np.transpose(label)
    Len = (len(Label))
    class_number = np.max(Label) + 1
    _Sample = np.zeros(np.int(class_number))
    while i < Len:
        if count < class_number and _Sample[count] < 15:
          _Sample[count] = _Sample[count] + 1
          address_test.append(address[i])
          label_test.append(Label[i])
          attribute_test.append(Attribute[i])
          i = i + 1
        else:
          while i < Len and Label[i] == count:
            if train_sample<50:
               address_train.append(address[i])
               label_train.append(Label[i])
               attribute_train.append(Attribute[i])
               train_sample=train_sample+1
            i = i + 1
          count = count + 1
          train_sample = 0
    return address_train, np.transpose(attribute_train),label_train, address_test, np.transpose(attribute_test),label_test
'''
def vgg_16(image,variables_dict):

   inputs=tf.cast(image,tf.float32)

   conv = tf.nn.conv2d(inputs, variables_dict['conv1_1_weights'], [1, 1, 1, 1], padding='SAME')


   bias = tf.nn.bias_add(conv, variables_dict['conv1_1_biases'])




   conv1_1 = tf.nn.relu(bias, name='conv1_1')


   conv = tf.nn.conv2d(conv1_1, variables_dict['conv1_2_weights'], [1, 1, 1, 1], padding='SAME')



   bias = tf.nn.bias_add(conv, variables_dict['conv1_2_biases'])



   conv1_2 = tf.nn.relu(bias, name='conv1_2')

   pool1=tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name= 'pool1')


   conv = tf.nn.conv2d(pool1, variables_dict['conv2_1_weights'], [1, 1, 1, 1], padding='SAME')

   bias = tf.nn.bias_add(conv, variables_dict['conv2_1_biases'])

   #bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
   conv2_1 = tf.nn.relu(bias, name='conv2_1')


   conv = tf.nn.conv2d(conv2_1, variables_dict['conv2_2_weights'], [1, 1, 1, 1], padding='SAME')

   bias = tf.nn.bias_add(conv, variables_dict['conv2_2_biases'])


  # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
   conv2_2 = tf.nn.relu(bias, name='conv2_2')

   pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool2')


   conv = tf.nn.conv2d(pool2, variables_dict['conv3_1_weights'], [1, 1, 1, 1], padding='SAME')

   bias = tf.nn.bias_add(conv, variables_dict['conv3_1_biases'])

   #bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
   conv3_1 = tf.nn.relu(bias, name='conv3_1')


   conv = tf.nn.conv2d(conv3_1, variables_dict['conv3_2_weights'], [1, 1, 1, 1], padding='SAME')

   bias = tf.nn.bias_add(conv, variables_dict['conv3_2_biases'])

   #bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
   conv3_2 = tf.nn.relu(bias, name='conv3_2')

   
   conv = tf.nn.conv2d(conv3_2, variables_dict['conv3_3_weights'], [1, 1, 1, 1], padding='SAME')

   bias = tf.nn.bias_add(conv, variables_dict['conv3_3_biases'])
  # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
   conv3_3 = tf.nn.relu(bias, name='conv3_3')



   pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool3')


   conv = tf.nn.conv2d(pool3, variables_dict['conv4_1_weights'], [1, 1, 1, 1], padding='SAME')

   bias = tf.nn.bias_add(conv, variables_dict['conv4_1_biases'])
   #bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
   conv4_1 = tf.nn.relu(bias, name='conv4_1')


   conv = tf.nn.conv2d(conv4_1, variables_dict['conv4_2_weights'], [1, 1, 1, 1], padding='SAME')

   bias = tf.nn.bias_add(conv, variables_dict['conv4_2_biases'])
   #bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
   conv4_2 = tf.nn.relu(bias, name='conv4_2')


   conv = tf.nn.conv2d(conv4_2, variables_dict['conv4_3_weights'], [1, 1, 1, 1], padding='SAME')

   bias = tf.nn.bias_add(conv, variables_dict['conv4_3_biases'])

   #bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
   conv4_3 = tf.nn.relu(bias, name='conv4_3')


   pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool4')

   conv = tf.nn.conv2d(pool4, variables_dict['conv5_1_weights'], [1, 1, 1, 1], padding='SAME')

   bias = tf.nn.bias_add(conv, variables_dict['conv5_1_biases'])
  # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
   conv5_1 = tf.nn.relu(bias, name='conv5_1')


   conv = tf.nn.conv2d(conv5_1, variables_dict['conv5_2_weights'], [1, 1, 1, 1], padding='SAME')

   bias = tf.nn.bias_add(conv, variables_dict['conv5_2_biases'])
  # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
   conv5_2 = tf.nn.relu(bias, name='conv5_2')

   conv = tf.nn.conv2d(conv5_2, variables_dict['conv5_3_weights'], [1, 1, 1, 1], padding='SAME')

   bias = tf.nn.bias_add(conv, variables_dict['conv5_3_biases'])
  # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
   conv5_3 = tf.nn.relu(bias, name='conv5_3')


   pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')

   shape = pool5.get_shape().as_list()

   dim = 1
   for d in shape[1:]:
     dim *= d
   x = tf.reshape(pool5, [-1, dim])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
   #variables_dict['fc6_weights'] = tf.contrib.layers.batch_norm(variables_dict['fc6_weights'], center=True, scale=True, is_training=phase)
  # variables_dict['fc6_biases'] = tf.contrib.layers.batch_norm(variables_dict['fc6_biases'], center=True, scale=True,is_training=phase)
   fc6 = tf.nn.bias_add(tf.matmul(x, variables_dict['fc6_weights']), variables_dict['fc6_biases'])
  # fc6  = tf.contrib.layers.batch_norm(fc6 , center=True, scale=True, is_training=phase)
   fc6 = tf.nn.relu(fc6, name='fc6')
  # dropout = tf.layers.dropout(inputs= fc7, rate=0.5, training=phase)
  # fc8 = tf.nn.bias_add(tf.matmul(dropout, variables_dict['fc8_weights']), variables_dict['fc8_biases'])

   fc7 = tf.nn.bias_add(tf.matmul(fc6, variables_dict['fc7_weights']), variables_dict['fc7_biases'])
   #fc8  = tf.contrib.layers.batch_norm(fc8 , center=True, scale=True, is_training=phase)
   return fc7
'''
def vgg_19(image,variables_dict):

   inputs=tf.cast(image,tf.float32)

   conv = tf.nn.conv2d(inputs, variables_dict['conv1_1_weights'], [1, 1, 1, 1], padding='SAME')


   bias = tf.nn.bias_add(conv, variables_dict['conv1_1_biases'])


   #bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)

   conv1_1 = tf.nn.relu(bias, name='conv1_1')


   conv = tf.nn.conv2d(conv1_1, variables_dict['conv1_2_weights'], [1, 1, 1, 1], padding='SAME')



   bias = tf.nn.bias_add(conv, variables_dict['conv1_2_biases'])


   #bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)

   conv1_2 = tf.nn.relu(bias, name='conv1_2')

   pool1=tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name= 'pool1')


   conv = tf.nn.conv2d(pool1, variables_dict['conv2_1_weights'], [1, 1, 1, 1], padding='SAME')

   bias = tf.nn.bias_add(conv, variables_dict['conv2_1_biases'])

   #bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
   conv2_1 = tf.nn.relu(bias, name='conv2_1')


   conv = tf.nn.conv2d(conv2_1, variables_dict['conv2_2_weights'], [1, 1, 1, 1], padding='SAME')

   bias = tf.nn.bias_add(conv, variables_dict['conv2_2_biases'])


  # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
   conv2_2 = tf.nn.relu(bias, name='conv2_2')

   pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool2')


   conv = tf.nn.conv2d(pool2, variables_dict['conv3_1_weights'], [1, 1, 1, 1], padding='SAME')

   bias = tf.nn.bias_add(conv, variables_dict['conv3_1_biases'])

   #bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
   conv3_1 = tf.nn.relu(bias, name='conv3_1')


   conv = tf.nn.conv2d(conv3_1, variables_dict['conv3_2_weights'], [1, 1, 1, 1], padding='SAME')

   bias = tf.nn.bias_add(conv, variables_dict['conv3_2_biases'])

   #bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
   conv3_2 = tf.nn.relu(bias, name='conv3_2')


   conv = tf.nn.conv2d(conv3_2, variables_dict['conv3_3_weights'], [1, 1, 1, 1], padding='SAME')

   bias = tf.nn.bias_add(conv, variables_dict['conv3_3_biases'])
  # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
   conv3_3 = tf.nn.relu(bias, name='conv3_3')


   conv = tf.nn.conv2d(conv3_3, variables_dict['conv3_4_weights'], [1, 1, 1, 1], padding='SAME')

   bias = tf.nn.bias_add(conv, variables_dict['conv3_4_biases'])
  # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
   conv3_4 = tf.nn.relu(bias, name='conv3_4')

   pool3 = tf.nn.max_pool(conv3_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool3')


   conv = tf.nn.conv2d(pool3, variables_dict['conv4_1_weights'], [1, 1, 1, 1], padding='SAME')

   bias = tf.nn.bias_add(conv, variables_dict['conv4_1_biases'])
   #bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
   conv4_1 = tf.nn.relu(bias, name='conv4_1')


   conv = tf.nn.conv2d(conv4_1, variables_dict['conv4_2_weights'], [1, 1, 1, 1], padding='SAME')

   bias = tf.nn.bias_add(conv, variables_dict['conv4_2_biases'])
   #bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
   conv4_2 = tf.nn.relu(bias, name='conv4_2')


   conv = tf.nn.conv2d(conv4_2, variables_dict['conv4_3_weights'], [1, 1, 1, 1], padding='SAME')

   bias = tf.nn.bias_add(conv, variables_dict['conv4_3_biases'])

   #bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
   conv4_3 = tf.nn.relu(bias, name='conv4_3')


   conv = tf.nn.conv2d(conv4_3, variables_dict['conv4_4_weights'], [1, 1, 1, 1], padding='SAME')

   bias = tf.nn.bias_add(conv, variables_dict['conv4_4_biases'])
   #bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
   conv4_4 = tf.nn.relu(bias, name='conv4_4')
   pool4 = tf.nn.max_pool(conv4_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool4')

   conv = tf.nn.conv2d(pool4, variables_dict['conv5_1_weights'], [1, 1, 1, 1], padding='SAME')

   bias = tf.nn.bias_add(conv, variables_dict['conv5_1_biases'])
  # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
   conv5_1 = tf.nn.relu(bias, name='conv5_1')


   conv = tf.nn.conv2d(conv5_1, variables_dict['conv5_2_weights'], [1, 1, 1, 1], padding='SAME')

   bias = tf.nn.bias_add(conv, variables_dict['conv5_2_biases'])
  # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
   conv5_2 = tf.nn.relu(bias, name='conv5_2')

   conv = tf.nn.conv2d(conv5_2, variables_dict['conv5_3_weights'], [1, 1, 1, 1], padding='SAME')

   bias = tf.nn.bias_add(conv, variables_dict['conv5_3_biases'])
  # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
   conv5_3 = tf.nn.relu(bias, name='conv5_3')


   conv = tf.nn.conv2d(conv5_3, variables_dict['conv5_4_weights'], [1, 1, 1, 1], padding='SAME')

   bias = tf.nn.bias_add(conv, variables_dict['conv5_4_biases'])
  # bias = tf.contrib.layers.batch_norm(bias, center=True, scale=True, is_training=phase)
   conv5_4 = tf.nn.relu(bias, name='conv5_4')
   pool5 = tf.nn.max_pool(conv5_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')

   shape = pool5.get_shape().as_list()

   dim = 1
   for d in shape[1:]:
     dim *= d
   x = tf.reshape(pool5, [-1, dim])
   fc6 = tf.nn.bias_add(tf.matmul(x, variables_dict['fc6_weights']), variables_dict['fc6_biases'])

   fc6 = tf.nn.relu(fc6, name='fc6')

   fc7 = tf.nn.bias_add(tf.matmul(fc6, variables_dict['fc7_weights']), variables_dict['fc7_biases'])

  # fc7 = tf.nn.relu(fc7, name='fc7')

  # fc8 = tf.nn.bias_add(tf.matmul(fc7, variables_dict['fc8_weights']), variables_dict['fc8_biases'])

   #fc8 = tf.nn.relu(fc8, name='fc8')
   '''
   conv = tf.nn.conv2d(pool5, variables_dict['fc6_weights'], [1, 1, 1, 1], padding='VALID')
   bias = tf.nn.bias_add(conv, variables_dict['fc6_biases'])

   fc6 = tf.nn.relu(bias, name='fc6')

   conv = tf.nn.conv2d(fc6, variables_dict['fc7_weights'], [1, 1, 1, 1], padding='SAME')
   bias = tf.nn.bias_add(conv, variables_dict['fc7_biases'])
   fc7 = tf.nn.relu(bias, name='fc7')

   conv = tf.nn.conv2d(fc7 , variables_dict['fc8_weights'], [1, 1, 1, 1], padding='SAME')
   bias = tf.nn.bias_add(conv, variables_dict['fc8_biases'])
   fc8 = tf.nn.relu(bias, name='fc8')
   fc8 = tf.squeeze(fc8, [1, 2], name='fc8')

   print(fc8.shape)
   return fc8
   '''

   return fc7

def _read_py_function(filenam,attribute, label):
    image_string = tf.read_file(filenam)
    image_decoded = tf.image.decode_jpeg(image_string,channels=3)
    image_resized = tf.image.resize_images(image_decoded , [224, 224])
    return image_resized,attribute, label
def Test_set(Batch_Size):

 #filenames, attribute, labels = Load_Dataset()
 address_train, attribute_train, label_train,filenames, attribute, labels=Load_Dataset()

 label=one_hot_label(labels,int(np.max(labels)+1))

 #inputt=[]
 #for i in range(len(filenames)):
#   inputt.append((filenames[i]))
 A=tf.constant(filenames)
 B=tf.constant(np.transpose(attribute))
 C=tf.constant(label)
 dataset = tf.contrib.data.Dataset.from_tensor_slices((A,B,C))
 dataset = dataset.map(_read_py_function)
 dataset = dataset.shuffle(buffer_size=len(label))
 dataset = dataset.batch(Batch_Size)

 return  dataset
def DataSet(Batch_Size):

 #filenames, attribute, labels = Load_Dataset()
 filenames, attribute, labels, address_test, attribute_test, label_test=Load_Dataset()

 label=one_hot_label(labels,int(np.max(labels)+1))
 #inputt=[]
 #for i in range(len(filenames)):
#   inputt.append((filenames[i]))
 A=tf.constant(filenames)
 B=tf.constant(np.transpose(attribute))
 C=tf.constant(label)
 dataset = tf.contrib.data.Dataset.from_tensor_slices((A,B,C))
 dataset = dataset.map(_read_py_function)
 dataset = dataset.shuffle(buffer_size=len(label))
 dataset = dataset.batch(Batch_Size)

 return  dataset


def one_hot_label(label,number_of_class):

    labels=np.zeros((len(label),number_of_class))

    for i in range (len(label)):
        labels[i,np.int(label[i])]=1.0
    return labels
def training_loop(images, labels,variables_dict):
    train_log_dir = "/home/fariborz/PycharmProjects"
    model_path = '/home/fariborz/PycharmProjects/Regresion/CHKPNT_ImageNETCasiaFace'
    if not tf.gfile.Exists(train_log_dir):
      tf.gfile.MakeDirs(train_log_dir)

   # predictions = vgg_16(tf.cast(images,tf.float32), num_classes=40, is_training=True, dropout_keep_prob=0.5, fc_conv_padding='VALID')
    # input_label=tf.constant(labels)
   # prediction= vgg_19(tf.cast(images,tf.float32), variables_dict)

    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=prediction))

   # opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    # Add Ops to the graph to minimize a cost by updating a list of variables.
    # "cost" is a Tensor, and the list of variables contains tf.Variable
    # objects.
   # opt_op = opt.minimize(cost,var_list=[variables_dict['fc6_weights'],variables_dict['fc6_biases'],variables_dict['fc7_weights'],variables_dict['fc7_biases'],variables_dict['fc8_weights'],variables_dict['fc8_biases']])
    #opt_op.run()
    # optimizer = tf.train.AdamOptimizer().minimize(cost)
    #loss = slim.losses.softmax_cross_entropy(predictions, labels)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001)
    # train_tensor = slim.learning.create_train_op(loss, optimizer)
    # init_fn = slim.assign_from_checkpoint_fn(model_path, variables_to_restore)
    #    predict=slim.learning.train(train_tensor, train_log_dir)
    #  restorer = tf.train.Saver()
    #    tf.summary.scalar('losses/total_loss', loss)
 #   return   prediction
variables_dict = {
"conv1_1_weights": tf.Variable(tf.random_normal([3, 3, 3, 64]),name="conv1_1_weights"),
"conv1_1_biases": tf.Variable(tf.zeros([64]), name="conv1_1_biases"),

"conv1_2_weights": tf.Variable(tf.random_normal([3, 3, 64, 64]),name="conv1_2_weights"),
"conv1_2_biases": tf.Variable(tf.zeros([64]), name="conv1_2_biases"),

"conv2_1_weights": tf.Variable(tf.random_normal([3, 3, 64, 128]),name="conv2_1_weights"),
"conv2_1_biases": tf.Variable(tf.zeros([128]), name="conv2_1_biases"),


"conv2_2_weights": tf.Variable(tf.random_normal([3, 3, 128, 128]),name="conv2_2_weights"),
"conv2_2_biases": tf.Variable(tf.zeros([128]), name="conv2_2_biases"),

"conv3_1_weights": tf.Variable(tf.random_normal([3, 3, 128, 256]),name="conv3_1_weights"),
"conv3_1_biases": tf.Variable(tf.zeros([256]), name="conv3_1_biases"),

"conv3_2_weights": tf.Variable(tf.random_normal([3, 3, 256, 256]),name="conv3_2_weights"),
"conv3_2_biases": tf.Variable(tf.zeros([256]), name="conv3_2_biases"),

"conv3_3_weights": tf.Variable(tf.random_normal([3, 3, 256, 256]),name="conv3_3_weights"),
"conv3_3_biases": tf.Variable(tf.zeros([256]), name="conv3_3_biases"),

"conv3_4_weights": tf.Variable(tf.random_normal([3, 3, 256, 256]),name="conv3_4_weights"),
"conv3_4_biases": tf.Variable(tf.zeros([256]), name="conv3_4_biases"),

"conv4_1_weights": tf.Variable(tf.random_normal([3, 3, 256, 512]),name="conv4_1_weights"),
"conv4_1_biases": tf.Variable(tf.zeros([512]), name="conv4_1_biases"),

"conv4_2_weights": tf.Variable(tf.random_normal([3, 3, 512, 512]),name="conv4_2_weights"),
"conv4_2_biases": tf.Variable(tf.zeros([512]), name="conv4_2_biases"),

"conv4_3_weights": tf.Variable(tf.random_normal([3, 3, 512, 512]),name="conv4_3_weights"),
"conv4_3_biases": tf.Variable(tf.zeros([512]), name="conv4_3_biases"),

"conv4_4_weights": tf.Variable(tf.random_normal([3, 3, 512, 512]),name="conv4_4_weights"),
"conv4_4_biases": tf.Variable(tf.zeros([512]), name="conv4_4_biases"),

"conv5_1_weights": tf.Variable(tf.random_normal([3, 3, 512, 512]),name="conv5_1_weights"),
"conv5_1_biases": tf.Variable(tf.zeros([512]), name="conv5_1_biases"),

"conv5_2_weights": tf.Variable(tf.random_normal([3, 3, 512, 512]),name="conv5_2_weights"),
"conv5_2_biases": tf.Variable(tf.zeros([512]), name="conv5_2_biases"),

"conv5_3_weights": tf.Variable(tf.random_normal([3, 3, 512, 512]),name="conv5_3_weights"),
"conv5_3_biases": tf.Variable(tf.zeros([512]), name="conv5_3_biases"),

"conv5_4_weights": tf.Variable(tf.random_normal([3, 3, 512, 512]),name="conv5_4_weights"),
"conv5_4_biases": tf.Variable(tf.zeros([512]), name="conv5_4_biases"),

"fc6_weights": tf.Variable(tf.random_normal([25088, 4096]),name="fc6_weights"),
"fc6_biases": tf.Variable(tf.zeros([4096]), name="fc6_biases"),

"fc7_weights": tf.Variable(tf.random_normal([4096, 200]),name="fc7_weights"),
"fc7_biases": tf.Variable(tf.zeros([200]), name="fc7_biases"),

#"fc8_weights": tf.Variable(tf.random_normal([4096, 200]),name="fc8_weights"),
#"fc8_biases": tf.Variable(tf.zeros([200]), name="fc8_biases")
}

'''
variables_dict = {
"conv1_1_weights": tf.Variable(tf.random_normal([3, 3, 3, 64]),name="conv1_1_weights"),
"conv1_1_biases": tf.Variable(tf.zeros([64]), name="conv1_1_biases"),

"conv1_2_weights": tf.Variable(tf.random_normal([3, 3, 64, 64]),name="conv1_2_weights"),
"conv1_2_biases": tf.Variable(tf.zeros([64]), name="conv1_2_biases"),

"conv2_1_weights": tf.Variable(tf.random_normal([3, 3, 64, 128]),name="conv2_1_weights"),
"conv2_1_biases": tf.Variable(tf.zeros([128]), name="conv2_1_biases"),


"conv2_2_weights": tf.Variable(tf.random_normal([3, 3, 128, 128]),name="conv2_2_weights"),
"conv2_2_biases": tf.Variable(tf.zeros([128]), name="conv2_2_biases"),

"conv3_1_weights": tf.Variable(tf.random_normal([3, 3, 128, 256]),name="conv3_1_weights"),
"conv3_1_biases": tf.Variable(tf.zeros([256]), name="conv3_1_biases"),

"conv3_2_weights": tf.Variable(tf.random_normal([3, 3, 256, 256]),name="conv3_2_weights"),
"conv3_2_biases": tf.Variable(tf.zeros([256]), name="conv3_2_biases"),

"conv3_3_weights": tf.Variable(tf.random_normal([3, 3, 256, 256]),name="conv3_3_weights"),
"conv3_3_biases": tf.Variable(tf.zeros([256]), name="conv3_3_biases"),



"conv4_1_weights": tf.Variable(tf.random_normal([3, 3, 256, 512]),name="conv4_1_weights"),
"conv4_1_biases": tf.Variable(tf.zeros([512]), name="conv4_1_biases"),

"conv4_2_weights": tf.Variable(tf.random_normal([3, 3, 512, 512]),name="conv4_2_weights"),
"conv4_2_biases": tf.Variable(tf.zeros([512]), name="conv4_2_biases"),

"conv4_3_weights": tf.Variable(tf.random_normal([3, 3, 512, 512]),name="conv4_3_weights"),
"conv4_3_biases": tf.Variable(tf.zeros([512]), name="conv4_3_biases"),



"conv5_1_weights": tf.Variable(tf.random_normal([3, 3, 512, 512]),name="conv5_1_weights"),
"conv5_1_biases": tf.Variable(tf.zeros([512]), name="conv5_1_biases"),

"conv5_2_weights": tf.Variable(tf.random_normal([3, 3, 512, 512]),name="conv5_2_weights"),
"conv5_2_biases": tf.Variable(tf.zeros([512]), name="conv5_2_biases"),

"conv5_3_weights": tf.Variable(tf.random_normal([3, 3, 512, 512]),name="conv5_3_weights"),
"conv5_3_biases": tf.Variable(tf.zeros([512]), name="conv5_3_biases"),


"fc6_weights": tf.Variable(tf.random_normal([25088,3500]),name="fc6_weights"),
"fc6_biases": tf.Variable(tf.zeros([3500]), name="fc6_biases"),
"fc7_weights": tf.Variable(tf.random_normal([3500, 200]),name="fc7_weights"),
"fc7_biases": tf.Variable(tf.zeros([200]), name="fc7_biases")
}
'''
phase = tf.placeholder(tf.bool)
#reader = pywrap_tensorflow.NewCheckpointReader('/home/fariborz/PycharmProjects/Regresion/CASIA_checkpoints/model.ckpt-118680')
reader = pywrap_tensorflow.NewCheckpointReader('/home/fariborz/PycharmProjects/Regresion/vgg16/vgg_19.ckpt')
#reader = pywrap_tensorflow.NewCheckpointReader('/tmp/checkPoint/model.ckpt')
var_to_shape_map = reader.get_variable_to_shape_map()
trained_variables = sorted(variables_dict)
tunned_weight = sorted(var_to_shape_map)
tag=tf.placeholder(tf.bool)
Number_of_epoch=200
iterator = DataSet(32).make_initializable_iterator()
next_element = iterator.get_next()
iterator_test = Test_set(32).make_initializable_iterator()
next_test_batch = iterator_test.get_next()
train_x = tf.placeholder(tf.float32, [None,224,224,3])#this part should be modified when you wanna evaluate your algorithm band by band
train_y = tf.placeholder(tf.float32,[None,200])

prediction =vgg_19(train_x,variables_dict)
#prediction =convolutional_net(train_x , variables_dict,phase)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=train_y, logits=prediction))
#Total_cost=Sparse_los(cost,variables_dict,100,1000)
opt = tf.train.AdamOptimizer(learning_rate=0.01)
#opt = tf.train.AdamOptimizer(learning_rate=0.01)
loss=opt.minimize(cost,var_list=[variables_dict["fc6_weights"],variables_dict["fc6_biases"],variables_dict["fc7_weights"],variables_dict["fc7_biases"]])
#loss=opt.minimize(Total_cost,var_list=[variables_dict["conv1_1_weights"],variables_dict["conv1_1_biases"],variables_dict["fc6_weights"],variables_dict["fc6_biases"],variables_dict["fc7_weights"],variables_dict["fc7_biases"]])
correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(train_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for key in range(32):
   # for key in range(len(variables_dict)):
        #print trained_variables[key]
       # print reader.get_tensor(tunned_weight[key]).shape
        sess.run(variables_dict[trained_variables[key]].assign(reader.get_tensor(tunned_weight[(key + 1)])))
       # sess.run(variables_dict[trained_variables[key]].assign(reader.get_tensor(trained_variables[key])))
    for epoch in range(Number_of_epoch):
        sess.run([iterator.initializer])
        num_steps = 0
        _total_acc = 0
        while True:
            try:

                _next_element= sess.run(next_element)


                _, _cost = sess.run([loss, cost],
                                          feed_dict={ train_x: _next_element[0],train_y: _next_element[2],phase:False})
                _acc = sess.run(accuracy,
                                          feed_dict={train_x: _next_element[0], train_y: _next_element[2], phase: False})
                _total_acc = _total_acc + _acc
                num_steps = num_steps + 1
                print (_acc)
              #  print (sess.run(show_time(variables_dict)))
               # S=sess.run(variables_dict['conv1_1_biases'])
             #   print S[0:5]

            except tf.errors.OutOfRangeError:
               #print('Epoch', epoch, 'completed out of', Number_of_epoch)
              # save_path = saver.save(sess, "/tmp/checkPoint/model.ckpt")
               print(epoch,' is completed')
               print('train acc: {}'.format(_total_acc / num_steps))
               break

        print ('test is started')

        num_steps = 0
        _total_acc = 0
        sess.run([iterator_test.initializer])
        while True:
            try:

                _next_element= sess.run(next_test_batch)

                _, _cost = sess.run([loss, cost],
                                          feed_dict={ train_x: _next_element[0],train_y: _next_element[2],phase:False})
                _acc = sess.run(accuracy,
                                          feed_dict={train_x: _next_element[0], train_y: _next_element[2], phase: False})
                _total_acc = _total_acc + _acc
                num_steps = num_steps + 1
                print (_acc)
            #    print (sess.run(show_time(variables_dict)))
               # S=sess.run(variables_dict['conv1_1_biases'])
             #   print S[0:5]

            except tf.errors.OutOfRangeError:
               #print('Epoch', epoch, 'completed out of', Number_of_epoch)
              # save_path = saver.save(sess, "/tmp/checkPoint/model.ckpt")
               print(epoch,' test is completed')
               print('test acc: {}'.format(_total_acc / num_steps))
               break
