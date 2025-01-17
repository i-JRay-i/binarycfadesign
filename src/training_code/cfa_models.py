"""
This Python file contains custom layers and models used in simultaneous CFA 
filter learning and demosaicing models. One custom layer is designed for 
Maximum Thresholding CFA learning algorithm. Another layer implements a Weighted 
SoftMax method from another study.

Written by: Cemre Ömer Ayna
"""

import tensorflow as tf
import numpy as np
import cv2

"""
Fixed filter designs that are used for comparison with the model.
"""

def show_filter(demosaicer, name="Filter", save=False):
    cv2_learned_filter = cv2.resize(demosaicer.astype("uint8"), (480,480), interpolation=cv2.INTER_AREA).astype("uint8");
    cv2.imshow("Filter", cv2_learned_filter);
    cv2.waitKey(0);
    if save == True:
        cv2.imwrite(name+".png", cv2_learned_filter);

def bayer_filter(filter_size):
    bayer = np.array([[[1,0,0],[0,1,0]],
                      [[0,1,0],[0,0,1]]]);
    size = round(filter_size/2);
    bayer = np.tile(bayer, (size,size,1));
    return bayer;

def lukac_filter(filter_size):
    lukac = np.array([[[0,1,0],[0,0,1]],
                      [[0,1,0],[1,0,0]],
                      [[0,0,1],[0,1,0]],
                      [[1,0,0],[0,1,0]]]);
    size = round(filter_size/2);
    lukac = np.tile(lukac, (int(size/2),size,1));
    return lukac;

def rgbw_filter(filter_size):
    rgbw = np.array([[[1,0,0,0],[0,1,0,0]],
                      [[0,0,0,1],[0,0,1,0]]]);
    size = round(filter_size/2);
    rgbw = np.tile(rgbw, (size,size,1));
    return rgbw;

def cfz_filter(filter_size):
    cfz = np.asarray([[[0,1,0,0],[0,0,1,0],[1,0,0,0],[1,0,0,0]],
                      [[0,0,1,0],[0,0,0,1],[1,0,0,0],[1,0,0,0]],
                      [[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]],
                      [[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]]]);
    size = round(filter_size/4);
    cfz = np.tile(cfz, (size,size,1));
    return cfz;

def custom_filter(filter_address):
    custom_filter = np.load(filter_address)[0];
    custom_filter = np.tile(custom_filter, (3,3,1));
    return custom_filter;
"""
Callbacks for incremental sigmoid weight value of the first layer during training.
"""
class SetAlphaPerBatch(tf.keras.callbacks.Callback):
    def __init__(self, name, gamma=0.000025, batch_size=128):
        super().__init__();
        self.name = name;
        self.gamma = gamma;
        self.bs = batch_size;
        
    def on_train_batch_begin(self, batch, logs=None):
        filter_layer = self.model.get_layer(self.name);
        filter_layer.set_time(filter_layer.get_time() + self.bs);
        new_alpha = 1 + (self.gamma*filter_layer.get_time())**2;
        filter_layer.set_alpha(new_alpha);
        #print("New alpha in batch: ", filter_layer.get_alpha());

class SetAlphaPerEpoch(tf.keras.callbacks.Callback):
    def __init__(self, name, gamma=0.000025, num_dataset=40000):
        super().__init__();
        self.name = name;
        self.gamma = gamma;
        self.ndata = num_dataset;
        
    def on_epoch_begin(self, epoch, logs=None):
        filter_layer = self.model.get_layer(self.name);
        filter_layer.set_time(self.ndata*epoch);
        new_alpha = 1 + (self.gamma*filter_layer.get_time())**2;
        filter_layer.set_alpha(new_alpha);
        #print("New alpha in epoch: ", filter_layer.get_alpha());

"""
Custom weighted sigmoid filter as described in the paper.

    input        = 24 x 24 x 4 (CBGR Image)
    filter block = 8 x 8 x 4   (CBGR Filter Block)
    full filter  = 24 x 24 x 4 (Full CBGR Filter)
    output       = 24 x 24 x 1 
"""
class WeightedSoftmaxFilter(tf.keras.layers.Layer):
    def __init__(self, filter_size, channels, alpha=1):
        self.alpha = alpha;
        self.t = 0;
        
        w_init = tf.random_uniform_initializer(minval=0.01, maxval=0.99);
        self.P = filter_size;
        self.w = tf.Variable(w_init(shape=(1, filter_size, filter_size, channels)));
        super(WeightedSoftmaxFilter, self).__init__();
    
    def build(self, input_shape):
        pass;
        
    def call(self, input_tensor):
        height, width = input_tensor.shape[1:3];
        w_alpha = self.alpha * self.w;
        w_sigm = tf.nn.softmax(w_alpha, axis=-1);
        
        softmax_filter = w_sigm;
        for col in range(int(width/self.P)-1):
            softmax_filter = tf.concat([softmax_filter, w_sigm], axis=2);
        for row in range(int(height/self.P)-1):
            filter_row = w_sigm;
            for col in range(int(width/self.P)-1):
                filter_row = tf.concat([filter_row, w_sigm], axis=2);
            softmax_filter = tf.concat([softmax_filter, filter_row], axis=1);
        
        input_filtered = tf.multiply(softmax_filter, input_tensor);
        return input_filtered;
    
    def get_alpha(self):
        return self.alpha;
    
    def set_alpha(self, new_alpha):
        self.alpha = new_alpha;
        
    def get_time(self):
        return self.t;
    
    def set_time(self, time):
        self.t = time;
        
"""
The CFA module which was described in Henz's paper.
"""
class LinearFilter(tf.keras.layers.Layer):
    def __init__(self, filter_size, channels):
        w_init = tf.random_uniform_initializer(minval=0, maxval=1);
        b_init = tf.random_uniform_initializer(minval=0, maxval=1);
        self.w = tf.Variable(w_init(shape=(1, filter_size, filter_size, channels)));
        self.b = tf.Variable(b_init(shape=(1, filter_size, filter_size, 1)));
        
        self.P = filter_size;
        self.C = channels;
        super(LinearFilter, self).__init__();
        
    def build(self, input_shape):
        pass;
    
    def call(self, input_tensor):
        height, width = input_tensor.shape[1:3];
        linear_filter = self.w;
        linear_filter = tf.nn.relu(linear_filter);
        bias = self.b;
        for col in range(int(width/self.P)-1):
            linear_filter = tf.concat([linear_filter, self.w], axis=2);
            bias = tf.concat([bias, self.b], axis=2);
        for row in range(int(height/self.P)-1):
            filter_row = self.w;
            bias_row = self.b;
            for col in range(int(width/self.P)-1):
                filter_row = tf.concat([filter_row, self.w], axis=2);
                bias_row = tf.concat([bias_row, self.b], axis=2);
            linear_filter = tf.concat([linear_filter, filter_row], axis=1);
            bias = tf.concat([bias, bias_row], axis=1);
            
        inputs_weighted = tf.math.multiply(input_tensor, linear_filter);
        inputs_summed = tf.math.reduce_sum(inputs_weighted, axis=3, keepdims=True);
        inputs_filtered = inputs_summed + bias;
        return [inputs_filtered, inputs_weighted];

"""
The function that sets the threshold value for the maximum threshold layer and
applies to the incoming weights. The function includes a custom gradient equal
to that of the identity function.
"""
@tf.custom_gradient
def threshold_weights(weights, threshold=0):
    bin_w = tf.where(weights < threshold,
                0*tf.ones_like(weights, dtype=tf.float32),
                1*tf.ones_like(weights, dtype=tf.float32));
    def grad(dw):
        return dw;
    return bin_w, grad;

"""
Custom maximum threshold filter.

    input        = 3P x 3P x C
    weights      = P x P x C
    full filter  = 3P x 3P x C
    output       = 3P x 3P x 1
"""
class MaxThresholdFilter(tf.keras.layers.Layer):
    def __init__(self, filter_size, channels, **kwargs):
        super(MaxThresholdFilter, self).__init__();
        self.w_init = tf.random_normal_initializer();
        self.P = filter_size;
        self.C = channels;

    def build(self, input_shape):
        self.w = self.add_weight(shape=(1, self.P, self.P, self.C),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 name="threshold_filter");

    def call(self, input_tensor):
        w_max = self.w - tf.math.reduce_max(self.w, axis=3, keepdims=True);
        w_max =  threshold_weights(w_max);

        height, width = input_tensor.shape[1:3];
        max_filter = w_max;
        for col in range(int(width/self.P)-1):
            max_filter = tf.concat([max_filter, w_max], axis=2);
        for row in range(int(height/self.P)-1):
            filter_row = w_max
            for col in range(int(width/self.P)-1):
                filter_row = tf.concat([filter_row, w_max], axis=2);
            max_filter = tf.concat([max_filter, filter_row], axis=1);

        input_filtered = tf.multiply(max_filter, input_tensor);
        return input_filtered;
    
