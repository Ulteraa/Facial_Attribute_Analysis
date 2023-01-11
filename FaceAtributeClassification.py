from fileinput import filename
from scipy.io.matlab.mio import loadmat
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import vgg_19
import numpy as np
import cv2
import tables
import h5py
import scipy.io
import glob
import os
import time
from tensorflow.python import pywrap_tensorflow
import inspect
def Load_Dataset():
    # Set up the data loading:

    # A=h5py.File('/home/fariborz/PycharmProjects/Regresion/lfw_att_40.mat')
    # B = tables.open_file('/home/fariborz/PycharmProjects/Regresion/lfw_att_40.mat')
    arrays = {}
    address = []
    # data = []
    f = h5py.File("/home/fariborz/PycharmProjects/Regresion/lfw_att_40.mat")
    for column in f['name']:
        row_data = []

        for row_number in range(len(column)):
            #        print ''.join(map(unichr, f[column[row_number]][:]))
            #        for p in range(len(f[column[row_number]][:])):
            #              if f[column[row_number]][p]==92:
            #               f[column[row_number]][p]=47
            #Temp=()
            row_data.append(''.join(map(unichr, f[column[row_number]][:])))
            # print row_data
        address.append(row_data)
        # print data
    label = f['label']
    for i in range(len(address)):
        address[i][0]='lfw/'+address[i][0]
    _row,_col=label.shape

    Label=np.transpose(label)

    #assert address.shape[0] == Label.shape[0]
   # for i in range(_row):
    #    for j in range(_col):
    #      Label[j][i]=label[i][j]
    return address[0:13120],Label[0:13120]
  #  return address, Label
    # print data[1]
    # print f['label'][:,1]
    # Dir="lfw/"+data[1]
    # print (f['label'][:,1])
    # List= glob.glob("/home/fariborz/PycharmProjects/Regresion/lfw/*/*.jpg")
    # image_string = cv2.imread(('lfw/'+(data[0][0])))
    # print image_string
    # a = si.loadmat('/home/fariborz/PycharmProjects/Regresion/LFW/lfw_Name_40.mat')
    # b = a['Name']                # type(b) <type 'numpy.ndarray'>
    # list_of_strings  = b.tolist()           # type(list_of_strings ) <type 'list'>

    # print list_of_strings[1]
def vgg_16(image,variables_dict,phase):

   inputs=tf.cast(image[0],tf.float32)

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

   '''
   fc7 = tf.nn.bias_add(tf.matmul(fc6, variables_dict['fc7_weights']), variables_dict['fc7_biases'])
   #fc7  = tf.contrib.layers.batch_norm(fc7 , center=True, scale=True, is_training=phase)
   fc7 = tf.nn.relu(fc7, name='fc7')
   '''
  # dropout = tf.layers.dropout(inputs= fc7, rate=0.5, training=phase)
  # fc8 = tf.nn.bias_add(tf.matmul(dropout, variables_dict['fc8_weights']), variables_dict['fc8_biases'])

   fc8 = tf.nn.bias_add(tf.matmul(fc6, variables_dict['fc8_weights']), variables_dict['fc8_biases'])
   #fc8  = tf.contrib.layers.batch_norm(fc8 , center=True, scale=True, is_training=phase)
   cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=image[1], logits=fc8))
   return fc8 ,image[1],cost
'''
def vgg_19(image,variables_dict,phase):

   inputs=tf.cast(image[0],tf.float32)

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
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
   #variables_dict['fc6_weights'] = tf.contrib.layers.batch_norm(variables_dict['fc6_weights'], center=True, scale=True, is_training=phase)
  # variables_dict['fc6_biases'] = tf.contrib.layers.batch_norm(variables_dict['fc6_biases'], center=True, scale=True,is_training=phase)
   fc6 = tf.nn.bias_add(tf.matmul(x, variables_dict['fc6_weights']), variables_dict['fc6_biases'])
  # fc6  = tf.contrib.layers.batch_norm(fc6 , center=True, scale=True, is_training=phase)
   fc6 = tf.nn.relu(fc6, name='fc6')


   fc7 = tf.nn.bias_add(tf.matmul(fc6, variables_dict['fc7_weights']), variables_dict['fc7_biases'])
   #fc7  = tf.contrib.layers.batch_norm(fc7 , center=True, scale=True, is_training=phase)
   fc7 = tf.nn.relu(fc7, name='fc7')
  # dropout = tf.layers.dropout(inputs= fc7, rate=0.5, training=phase)
  # fc8 = tf.nn.bias_add(tf.matmul(dropout, variables_dict['fc8_weights']), variables_dict['fc8_biases'])

   fc8 = tf.nn.bias_add(tf.matmul(fc7, variables_dict['fc8_weights']), variables_dict['fc8_biases'])
   #fc8  = tf.contrib.layers.batch_norm(fc8 , center=True, scale=True, is_training=phase)
   fc8 = tf.nn.relu(fc8, name='fc8')
   cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=image[1], logits=fc8))
   return fc8 ,image[1],cost

'''
def _read_py_function(filenam, label):
    image_string = tf.read_file(filenam)
    image_decoded = tf.image.decode_jpeg(image_string,channels=3)

    image_resized = tf.image.resize_images(image_decoded , [224, 224])


    return image_resized, label
def DataSet(Batch_Size,Attribute_ID):

 filenames, labels = Load_Dataset()


 label=one_hot_label(labels[:,Attribute_ID],int(np.max(labels[:,Attribute_ID])+1))

 inputt=[]
# label=[]


 for i in range(len(filenames)):
   inputt.append(filenames[i][0])
  # label.append(labels[i][Attribute_ID])
 #print label

 #image_decoded = cv2.imread(filenames[10][0])
 #print image_decoded
 #AA=cv2.imread(inputt[1])
 #print AA.shape
 #print(len(filenames),len(labels[1][:]))
 A=tf.constant(inputt)
 B=tf.constant(label)
 dataset = tf.contrib.data.Dataset.from_tensor_slices((A,B))
 dataset = dataset.map(_read_py_function)
 dataset = dataset.shuffle(buffer_size=len(label))
 dataset = dataset.batch(Batch_Size)

 return  dataset
# dataset = dataset.repeat(1)

 #dataset = dataset.batch(Batch_Size)
 #return dataset
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
'''
variables_dict = {
"conv1_1_weights": tf.Variable(tf.random_normal([3, 3, 3, 64]),name="conv1_1_weights"),
"conv1_1_biases": tf.Variable(tf.zeros([64]), name="conv1_1_biases"),

"conv1_2_weights": tf.Variable(tf.random_normal([3, 3, 64, 64]),name="conv1_2_weights"),
"conv1_2_biases": tf.Variable(tf.zeros([64]), name="conv1_2_biases"),

"conv2_1_weights": tf.Variable(tf.random_normal([3, 3, 64, 128]),name="conv1_2_weights"),
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

"fc7_weights": tf.Variable(tf.random_normal([4096, 4096]),name="fc7_weights"),
"fc7_biases": tf.Variable(tf.zeros([4096]), name="fc7_biases"),

"fc8_weights": tf.Variable(tf.random_normal([4096, 2]),name="fc8_weights"),
"fc8_biases": tf.Variable(tf.zeros([2]), name="fc8_biases")
}
'''
variables_dict = {
"conv1_1_weights": tf.Variable(tf.random_normal([3, 3, 3, 64]),name="conv1_1_weights"),
"conv1_1_biases": tf.Variable(tf.zeros([64]), name="conv1_1_biases"),

"conv1_2_weights": tf.Variable(tf.random_normal([3, 3, 64, 64]),name="conv1_2_weights"),
"conv1_2_biases": tf.Variable(tf.zeros([64]), name="conv1_2_biases"),

"conv2_1_weights": tf.Variable(tf.random_normal([3, 3, 64, 128]),name="conv1_2_weights"),
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


"fc6_weights": tf.Variable(tf.random_normal([25088,512]),name="fc6_weights"),
"fc6_biases": tf.Variable(tf.zeros([512]), name="fc6_biases"),
"fc8_weights": tf.Variable(tf.random_normal([512, 2]),name="fc8_weights"),
"fc8_biases": tf.Variable(tf.zeros([2]), name="fc8_biases")
}
'''
"fc7_weights": tf.Variable(tf.random_normal([8, 8]),name="fc7_weights"),
"fc7_biases": tf.Variable(tf.zeros([8]), name="fc7_biases"),

"fc8_weights": tf.Variable(tf.random_normal([8, 2]),name="fc8_weights"),
"fc8_biases": tf.Variable(tf.zeros([2]), name="fc8_biases")

}

'''
phase = tf.placeholder(tf.bool)
#reader = pywrap_tensorflow.NewCheckpointReader('/home/fariborz/PycharmProjects/Regresion/CHKPNT_ImageNETCasiaFace/model.ckpt-118680')
reader = pywrap_tensorflow.NewCheckpointReader('/home/fariborz/PycharmProjects/Regresion/vgg16/vgg_16.ckpt')

var_to_shape_map = reader.get_variable_to_shape_map()
trained_variables = sorted(variables_dict)
tunned_weight = sorted(var_to_shape_map)
Attribute_ID=39
Data =DataSet(32,Attribute_ID)
Number_of_epoch=100
iterator = Data.make_initializable_iterator()
next_element = iterator.get_next()
prediction,_real_label,cost=vgg_16(next_element, variables_dict,phase)
#cost=tf.nn.softmax_cross_entropy_with_logits
#cost= tf.contrib.losses.sum_of_squares(prediction, _real_label)
#cost=tf.losses.mean_squared_error(prediction, _real_label)
#train_op1 = tf.train.GradientDescentOptimizer(0.0001).minimize(cost, var_list=[variables_dict['fc6_weights'],variables_dict['fc6_biases']])
#train_op2 = tf.train.GradientDescentOptimizer(0.001).minimize(cost, var_list=[variables_dict['fc7_weights'],variables_dict['fc7_biases'],variables_dict['fc8_weights'],variables_dict['fc8_biases']])
#opt_op = tf.group(train_op1, train_op2)
#opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)
opt=tf.train.AdamOptimizer(learning_rate=0.01)
opt_op = opt.minimize(cost,var_list=[variables_dict['fc6_weights'],variables_dict['fc6_biases'],variables_dict['fc8_weights'],variables_dict['fc8_biases']])
correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(_real_label, 1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for key in range(26):
        #Template=reader.get_tensor(tunned_weight[(key + 1)])
       # print Template.shape
        sess.run(variables_dict[trained_variables[key]].assign(reader.get_tensor(tunned_weight[(key + 1)])))
 #   sess.run(variables_dict[trained_variables[32]].assign(reader.get_tensor(tunned_weight[(33)])))
    '''
    X=reader.get_tensor(tunned_weight[(28)])
    x = tf.reshape(X, [25088, 4096])
    sess.run(variables_dict[trained_variables[27]].assign(x))
    sess.run(variables_dict[trained_variables[28]].assign(reader.get_tensor(tunned_weight[29])))
    Y = reader.get_tensor(tunned_weight[(30)])
    y = tf.reshape(Y, [4096, 4096])
    sess.run(variables_dict[trained_variables[29]].assign(y))
    '''
    '''
    sess.run(variables_dict[trained_variables[34]].assign(reader.get_tensor(tunned_weight[(35)])))
    sess.run(variables_dict[trained_variables[35]].assign(reader.get_tensor(tunned_weight[(36)])))
    '''
    for epoch in range(Number_of_epoch):
      sess.run([iterator.initializer])

      Acr=0; epoch_loss=0; count=0;test=0
      while True:
          count=count+1
          try:
              if count<300:

                 sess.run(opt_op)

               # print 'Fuck you'

                  #epoch_loss += c

              else:

                test=test+1
                Acr=Acr+sess.run(accuracy, feed_dict={phase: False})

                 # print('Accuracy:', sess.run(accuracy, feed_dict={phase: False}))
              # print 'batch was trained!'
          except tf.errors.OutOfRangeError:
              print('Epoch', epoch, 'completed out of', Number_of_epoch)
              print('Accuracy:', (Acr/(test-1)))
              break
'''
    while True:
        try:
            #print(sess.run(Training_Loop(next_element[0][:], next_element[1][:]).eval()))
        except tf.errors.OutOfRangeError:
            break
'''