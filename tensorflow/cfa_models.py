"""
This Python file contains custom layers and models used in simultaneous CFA 
filter learning and demosaicing models. One custom layer is designed for 
Maximum Thresholding CFA learning algorithm. Other layers implements Weighted 
SoftMax, Linear filter, and Probability Mask methods from other studies.

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

# Custom filter stripped off from a CFA learning module
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

"""
Probability mask implementation
Each weight is passed into a sigmoid to convert them to a probability representation.

    Wei = P x P x C number of weights initialized with random uniform initialization.
    Sig = P x P x C probability representations after sigmoid operation.
"""
class ProbMask(tf.keras.layers.Layer):
    def __init__(self, slope=5, filter_size=8, channels=4, **kwargs):
        self.slope = slope;
        self.P = filter_size;
        
        w_init = tf.random_uniform_initializer(minval=0.01, maxval=0.99);
        self.w = tf.Variable(w_init(shape=(1, filter_size, filter_size, channels)));
        self.w = tf.Variable(- tf.math.log(1. / self.w - 1.) / self.slope);
        super(ProbMask, self).__init__(**kwargs);
    
    def build(self, input_shape):
        super(ProbMask, self).build(input_shape);
    
    def call(self, input_tensor):
        height, width = input_tensor.shape[1:3];
        weights = self.w;
        return tf.sigmoid(self.slope * weights);

"""
The layer that rescales the weights given a sparsity level in probability map.
    Sig = P x P x C probability mask as input.
    Sig_Vec = P*P x C probability mask reshaped as vector.
    Prb_Vec = P*P x C probability mask rescaled with sparsity enforcement.
    Prb = P x P x C rescaled probability mask reshaped back to tensor.
"""

class RescaleProbMask(tf.keras.layers.Layer):
    def __init__(self, sparsity, **kwargs):
        self.alpha = sparsity;
        super(RescaleProbMask, self).__init__(**kwargs);
    
    def build(self, input_shape):
        self.height = input_shape[1];
        self.width = input_shape[2];
        self.num = input_shape[1] * input_shape[2];
        self.ch = input_shape[-1];
        super(RescaleProbMask, self).build(input_shape);
    
    def call(self, input_tensor):
        input_vec = tf.keras.layers.Reshape((self.num, -1))(input_tensor);
        prob_map = self.force_sparsity(input_vec[:,tf.newaxis,0,:], alpha=self.alpha);
        for i in range(self.num - 1):
            c = self.force_sparsity(input_vec[:,tf.newaxis,i+1,:], alpha=self.alpha);
            prob_map = tf.concat([prob_map,c], axis=1);
        prob_map = tf.keras.layers.Reshape((self.height, self.width, -1))(input_tensor);
        return prob_map;
    
    def force_sparsity(self, pixel, alpha):
        p = tf.math.reduce_mean(pixel);
        beta = (1 - alpha) / (1 - p);
        le = tf.cast(tf.greater_equal(p, alpha), tf.float32);
        return le * pixel * alpha / p + (1 - le) * (1 - beta * (1 - pixel));

"""
    Prb = P x P x C Probability mask as input.
    Trsh = P x P x C Uniform random threshold used in training to guarantee random walk.
    Rnd = P x P x C New probability mask passed through another sigmoid.
"""
class ThresholdProbMask(tf.keras.layers.Layer):
    def __init__(self, slope = 12, **kwargs):
        self.slope = None
        if slope is not None:
            self.slope = tf.Variable(slope, dtype=tf.float32);
        super(ThresholdProbMask, self).__init__(**kwargs);
    
    def build(self, input_shape):
        super(ThresholdProbMask, self).build(input_shape);
    
    def call(self, input_tensor):
        input_shape = tf.shape(input_tensor);
        thresh = tf.random.uniform(input_shape, minval=0.0, maxval=1.0, dtype='float32');
        if self.slope is not None:
            return tf.sigmoid(self.slope * (input_tensor-thresh));
        else:
            return input_shape > thresh;

"""
The layer that performs masking operation.

    input_image:    3*P x 3*P x C
    weights:        P x P x C

    weights_scaled: 3*P x 3*P x C
    image_masked:   3*P x 3*P x C
    image_merged:   3*P x 3*P x 1
"""
class UnderSample(tf.keras.layers.Layer):
    def __init__(self):
        super(UnderSample, self).__init__();
    
    def build(self, input_shape):
        super(UnderSample, self).build(input_shape);
    
    def call(self, inputs):
        input_image = inputs[0];
        weights = inputs[1];
        height = int(input_image.shape[1]/weights.shape[1]);
        width = int(input_image.shape[2]/weights.shape[2]);
        weights_scaled = self.scale_weights(weights, height, width);
            
        image_masked = tf.multiply(input_image, weights_scaled);
        image_merged = tf.reduce_sum(image_masked, axis=-1, keepdims=True);
        return image_merged;
    
    def scale_weights(self, weights, height, width):
        weights_sc = weights;
        for col in range(width-1):
            weights_sc = tf.concat([weights_sc, weights], axis=2);
        for row in range(height-1):
            filter_row = weights;
            for col in range(width-1):
                filter_row = tf.concat([filter_row, weights], axis=2);
            weights_sc = tf.concat([weights_sc, filter_row], axis=1);
        return weights_sc;

class ProbMaskFilter(tf.keras.layers.Layer):
    def __init__(self, pmask_slope, sample_slope, sm_filter_size, channel, sparsity, **kwargs):
        super(ProbMaskFilter, self).__init__();
        self.prob_mask = ProbMask(name='prob_mask', slope=pmask_slope, filter_size=sm_filter_size, channels=channel);
        self.mask_rescale = RescaleProbMask(name='prob_mask_scaled', sparsity=sparsity);
        self.thresh_mask = ThresholdProbMask(name='sampled_image', slope=sample_slope);
        self.undersample = UnderSample(name='under_sample_kspace');
        
    def build(self, input_shape):
        super(ProbMaskFilter, self).build(input_shape);

    def call(self, inputs):
        prob_mask_tensor = self.prob_mask(inputs);
        prob_mask_tensor_resc = self.mask_rescale(prob_mask_tensor);
        thresh_mask =  self.thresh_mask(prob_mask_tensor_resc);
        filtered_image = self.undersample([inputs, thresh_mask]);
        return filtered_image;
