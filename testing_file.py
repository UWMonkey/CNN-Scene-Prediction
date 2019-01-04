import tensorflow as tf 
import pipeline_clean 
import numpy as np 
import os

def inference(images,batch_size,unique_class):
# tf.reset_default_graph()
# image_placeholder =tf.placeholder(tf.float32,shape = (batch_size,IMAGE_PIXEL))
# shape = [kernal size, kernal size, channels, kernal numbers]
####  1st convolution layer name = 'conv1a' ###############
    with tf.variable_scope('conv1a', reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable('weights',
                                    shape=[3,3,3,16],
                                    dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                    shape = [16],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images,weights,strides=[1,1,1,1],padding='SAME')  
        pre_activation = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.relu(pre_activation,name = scope.name)
    ####  1st pooling layer name = 'pooling1_lrn' ###############
    with tf.variable_scope('pooling1_lrn',reuse=tf.AUTO_REUSE) as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1],strides=[1,2,2,1],
                                padding='SAME',name='pooling1')
        norm1 = tf.nn.lrn(pool1,depth_radius=4, bias=1.0,alpha=0.001/9.0,
                            beta=0.75,name="norm1") #cifar10 setting...need to explore

    ####  2st convolution layer name = 'conv2' ###############
    with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable('weights',
                                    shape=[3,3,16,16],
                                    dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                    shape = [16],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1,weights,strides=[1,1,1,1],padding='SAME')  
        pre_activation = tf.nn.bias_add(conv,biases)
        conv2 = tf.nn.relu(pre_activation,name = 'conv2')

    ####  2st pooling layer name = 'pooling2_lrn' ###############
    with tf.variable_scope('pooling2_lrn',reuse=tf.AUTO_REUSE) as scope:
        norm2 = tf.nn.lrn(conv2,depth_radius=4, bias=1.0,alpha=0.001/9.0,
                            beta=0.75,name="norm1") #cifar10 setting...need to explore
        pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1],strides=[1,1,1,1],
                                padding='SAME',name='pooling2')

    #### 1st fully connected layer #############################
    #for simplicity lets use 128 neurons
    with tf.variable_scope('local3', reuse=tf.AUTO_REUSE) as scope:
        #reshape = tf.reshape(pool2, shape = [batch_size,-1]) 
        #dim = reshape.get_shape()[1].value #readjsut this to get shape = [dim,128]
        shape = pool2.get_shape().as_list()
        dim = np.prod(shape[1:])
        reshape = tf.reshape(pool2, [-1, dim])
        weights = tf.get_variable('weights',
                                    shape=[dim,128],
                                    dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32)) 
        biases = tf.get_variable('biases',
                                    shape = [128],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape,weights)+biases, name = scope.name)
        #[64,4096]*[64,128] <----matmul
    #### 2st fully connected layer #############################
    #for simplicity lets use 128 neurons
    with tf.variable_scope('local4', reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable('weights',shape=[128,128],
                                    dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32)) 
        biases = tf.get_variable('biases',
                                    shape = [128],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3,weights)+biases, name = 'local4')


    #### output layer softmax ##################################
    with tf.variable_scope('softmax', reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable('softmax_linear',
                                    shape=[128,unique_class],
                                    dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32)) 
        biases = tf.get_variable('biases',
                                    shape = [unique_class],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4,weights),biases,name = 'softmax_linear')
    return softmax_linear

def losses(logits, labels):
    with tf.variable_scope('loss', reuse=tf.AUTO_REUSE) as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                        (logits=logits, labels=labels, name = 'xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name = 'loss')
        tf.summary.scalar(scope.name+'/loss',loss)
    return loss

def training(loss,learning_rate):
    with tf.name_scope('optimizer'):
        my_global_step = tf.Variable(0, name = 'global_step', trainable=False)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step = my_global_step)
    return train_op

def evaluation(logits,labels):
    with tf.variable_scope('accuracy', reuse = tf.AUTO_REUSE) as scope:
        correct = tf.nn.in_top_k(logits,labels,1)
        correct = tf.cast(correct,tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy',accuracy)
    return accuracy 

test_dir = "C://Users//KG//Desktop//MMAI 894//Project//Images//" #kevin's desktop
save_dir = "C://Users//KG//Desktop//MMAI 894//Project//save_workspace//" #kevin's desktop
image_list, label_list = pipeline_clean.get_file(test_dir)
image_tr,image_test = pipeline_clean.create_train_test(image_list)
label_tr,label_test = pipeline_clean.create_train_test(label_list)
batch_size = 16
tr_data = pipeline_clean.input_fn(image_tr,label_tr,batch_size)
test_data = pipeline_clean.input_fn(image_test,label_test,batch_size)

#def train():

images = tr_data['images']
labels = tr_data['labels']
init=tr_data['iterator_init_op']
    #Requirement for inference
unique_class = len(set(label_list))
learning_rate = 0.05
MAX_STEP = 1000
    #shape = [None, 64,64,3] <---this is what it is working with, width and height need change in pipeline
logits = inference(images,batch_size,unique_class)
loss = losses(logits,labels)

my_global_step=tf.Variable(0,name = 'global_step',trainable=False)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss,global_step = my_global_step)
saver = tf.train.Saver(tf.global_variables())
summary_op= tf.summary.merge_all()


init_var=tf.global_variables_initializer()
init_l = tf.local_variables_initializer()
init=tr_data['iterator_init_op']
sess=tf.Session()
sess.run(init)
sess.run(init_var)
sess.run(init_l)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess = sess, coord = coord)
summary_writer = tf.summary.FileWriter(save_dir,sess.graph)
try:
    for step in np.arange(MAX_STEP):
        if coord.should_stop():
            break
        _, loss_value = sess.run([train_op, loss])
        if step % 50==0:
            print("step: %d, loss: %4f" % (step,loss_value))
        if step % 100==0:
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)
        if step % 2000==0 or (step + 1) ==MAX_STEP:
            checkpoint_path = os.path.join(save_dir,'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)    
except tf.errors.OutOfRangeError:
    print("Done Training -- epoch limit reached")
finally:
    coord.request_stop()
coord.join(threads)
sess.close()







