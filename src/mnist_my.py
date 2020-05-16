from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def main(data_dir):
    mnist = input_data.read_data_sets(data_dir)
    
    with tf.Graph().as_default():
        images_placeholder = tf.placeholder(tf.float32, [None, 784])
        labels_placeholder = tf.placeholder(tf.int64, [None])
        
        logits = deepnet(images_placeholder)
        loss = get_loss(labels_placeholder, logits)
        train_op = train_step(loss)
        accuracy = get_accuracy(labels_placeholder, logits)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for _ in range(1000):
                image_batch, label_batch = mnist.train.next_batch(100)
                sess.run(train_op, feed_dict={images_placeholder: image_batch, 
                                              labels_placeholder: label_batch})
                
            accuracy_ = sess.run(accuracy, feed_dict={images_placeholder: mnist.test.images, 
                                                      labels_placeholder: mnist.test.labels})
            print("Accuracy: %2.3f\n" % accuracy_)

def deepnet(images):
    weights = tf.Variable(tf.zeros([784, 10]))
    biases = tf.Variable(tf.zeros([10]))
    logits = tf.matmul(images, weights) + biases
    return logits

def get_loss(labels, logits):
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    return tf.reduce_mean(cross_entropy)

def train_step(loss):
    train_op = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    return train_op

def get_accuracy(labels, logits):
    correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

if __name__ == '__main__':
    data_dir = '../data/'
    main(data_dir)
    print("____End____")































