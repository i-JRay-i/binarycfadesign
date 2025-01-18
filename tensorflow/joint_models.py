"""
Written by: Cemre Ömer Ayna

References: 
    
"""
import tensorflow as tf
import cfa_models as cfa
import demosaicing_models as dm
from math import exp, log

# Defining a dynamic learning rate with a scheduler function.
def scheduler(epoch, lr):
    lr_new = exp(epoch * (log(0.00001) - log(0.001)) / 99 + log(0.001));
    return lr_new;

class SaveEpochs(tf.keras.callbacks.Callback):
    def __init__(self, **kwargs):
        super().__init__();
        
    def on_epoch_begin(self, epoch, logs=None):
        epoch_checkpoint = [1, 2, 3, 4, 5, 8, 10, 15, 20, 30, 40, 50];
        if epoch in epoch_checkpoint:
            self.model.save("cfa_demosaicer_epoch_{}.keras".format(epoch));

""" Joint Models with Weighted Softmax Filter (Chakrabarti's)"""
# Weighted Softmax Filter + Chakrabarti's demosaicer
def WeightedSoftmax_Chak(imheight, imwidth, channel, sm_filter_size, gamma, proposals_num, rs_filter_num, batch_size, input_num):
    input_image = tf.keras.Input(shape=(imheight, imwidth, channel));
    filtered_image = cfa.WeightedSoftmaxFilter(filter_size = sm_filter_size,
                                           channels = channel)(input_image);
    filtered_image_merged = tf.reduce_sum(filtered_image, axis=-1, keepdims=True);
    output = dm.demosaicer_chak(filtered_image_merged, sm_filter_size, proposals_num, rs_filter_num);
    cfa_model = tf.keras.Model(inputs=input_image, outputs=output);
    name_filter = cfa_model.layers[1].name;
    cfa_callbacks = [cfa.SetAlphaPerBatch(name=name_filter, gamma=gamma, batch_size=batch_size),
                     cfa.SetAlphaPerEpoch(name=name_filter, gamma=gamma, num_dataset=input_num)];
    return [cfa_model, cfa_callbacks];

# Weighted Softmax Filter + Henz's demosaicer
def WeightedSoftmax_Henz(imheight, imwidth, channel, sm_filter_size, gamma, batch_size, input_num):
    input_image = tf.keras.Input(shape=(imheight, imwidth, channel));
    filtered_image = cfa.WeightedSoftmaxFilter(filter_size = sm_filter_size,
                                           channels = channel)(input_image);
    output = dm.demosaicer_henz(filtered_image);
    cfa_model = tf.keras.Model(inputs=input_image, outputs=output);
    name_filter = cfa_model.layers[1].name;
    cfa_callbacks = [cfa.SetAlphaPerBatch(name=name_filter, gamma=gamma, batch_size=batch_size),
                     cfa.SetAlphaPerEpoch(name=name_filter, gamma=gamma, num_dataset=input_num)];
    return [cfa_model, cfa_callbacks];

# Weighted Softmax Filter + de Gioia's demosaicer
def WeightedSoftmax_deGioia(imheight, imwidth, channel, sm_filter_size, demos_filter_num, gamma, batch_size, input_num):
    input_image = tf.keras.Input(shape=(imheight, imwidth, channel));
    filtered_image = cfa.WeightedSoftmaxFilter(filter_size = sm_filter_size,
                                           channels = channel)(input_image);
    output = dm.demosaicer_degiogia(filtered_image, demos_filter_num);
    cfa_model = tf.keras.Model(inputs=input_image, outputs=output);
    name_filter = cfa_model.layers[1].name;
    cfa_callbacks = [cfa.SetAlphaPerBatch(name=name_filter, gamma=gamma, batch_size=batch_size),
                     cfa.SetAlphaPerEpoch(name=name_filter, gamma=gamma, num_dataset=input_num)];
    return [cfa_model, cfa_callbacks];

# Weighted Softmax Filter + Custom demosaicer
def WeightedSoftmax_Custom(imheight, imwidth, channel, sm_filter_size, gamma, batch_size, input_num):
    input_image = tf.keras.Input(shape=(imheight, imwidth, channel));
    filtered_image = cfa.WeightedSoftmaxFilter(filter_size = sm_filter_size,
                                           channels = channel)(input_image);
    [proxy, inter_1, inter_2, inter_3, output_image] = dm.demosaicer_custom(filtered_image);
    cfa_model = tf.keras.Model(inputs=input_image, outputs=output_image);
    name_filter = cfa_model.layers[1].name;
    cfa_callbacks = [cfa.SetAlphaPerBatch(name=name_filter, gamma=gamma, batch_size=batch_size),
                     cfa.SetAlphaPerEpoch(name=name_filter, gamma=gamma, num_dataset=input_num)];
    return [cfa_model, cfa_callbacks];

""" Joint Models with Linear Filter (Henz's)"""
# Linear Filter + Chakrabarti's demosaicer
def Linear_Chak(imheight, imwidth, channel, filter_size, proposals_num, rs_filter_num):
    input_image = tf.keras.Input(shape=(imheight, imwidth, channel));
    [filtered_image, weighted_image] = cfa.LinearFilter(filter_size = filter_size, channels = channel)(input_image);
    filtered_image_merged = tf.reduce_sum(filtered_image, weighted_image, axis=-1, keepdims=True);
    output = dm.demosaicer_chak(filtered_image_merged, filter_size, proposals_num, rs_filter_num);
    cfa_model = tf.keras.Model(inputs=input_image, outputs=output);         # The callback object for the learning rate
    callback_lr = tf.keras.callbacks.LearningRateScheduler(scheduler);
    cfa_callbacks = [callback_lr];
    return [cfa_model, cfa_callbacks];

# Linear Filter + Henz's demosaicer
def Linear_Henz(imheight, imwidth, channel, filter_size):
    input_image = tf.keras.Input(shape=(imheight, imwidth, channel));
    [filtered_image, weighted_image] = cfa.LinearFilter(filter_size = filter_size, channels = channel)(input_image);
    output = dm.demosaicer_henz(filtered_image, weighted_image);
    cfa_model = tf.keras.Model(inputs=input_image, outputs=output);
    callback_lr = tf.keras.callbacks.LearningRateScheduler(scheduler);      # The callback object for the learning rate
    cfa_callbacks = [callback_lr];
    return [cfa_model, cfa_callbacks];

# Maximum Thresholding Filter + de Gioia's demosaicer
def Linear_deGioia(imheight, imwidth, channel, filter_size, demos_filter_num):
    input_image = tf.keras.Input(shape=(imheight, imwidth, channel));
    filtered_image = cfa.LinearFilter(filter_size = filter_size, channels = channel)(input_image);
    output = dm.demosaicer_degiogia(filtered_image, demos_filter_num);
    cfa_model = tf.keras.Model(inputs=input_image, outputs=output);
    callback_lr = tf.keras.callbacks.LearningRateScheduler(scheduler);      # The callback object for the learning rate
    cfa_callbacks = [callback_lr];
    return [cfa_model, cfa_callbacks];

# Maximum Thresholding Filter + Custom demosaicer 
def Linear_Custom(imheight, imwidth, channel, filter_size):
    input_image = tf.keras.Input(shape=(imheight, imwidth, channel));
    filtered_image = cfa.LinearFilter(filter_size = filter_size, channels = channel)(input_image);
    [proxy, inter_1, inter_2, inter_3, output_image] = dm.demosaicer_custom(filtered_image);
    cfa_model = tf.keras.Model(inputs=input_image, outputs=output_image);
    callback_lr = tf.keras.callbacks.LearningRateScheduler(scheduler);      # The callback object for the learning rate
    cfa_callbacks = [callback_lr];
    return [cfa_model, cfa_callbacks];

""" Joint Models with Probability Mask Filter"""
# Probability Mask Filter + Chakrabarti's demosaicer
def ProbMask_Chak(imheight, imwidth, channel, filter_size, pmask_slope, sample_slope, sparsity, proposals_num, rs_filter_num):
    input_image = tf.keras.Input(shape=(imheight, imwidth, channel));
    filtered_image = cfa.ProbMaskFilter(pmask_slope, sample_slope, filter_size, channel, sparsity)(input_image);
    filtered_image_merged = tf.reduce_sum(filtered_image, axis=-1, keepdims=True);
    output = dm.demosaicer_chak(filtered_image_merged, filter_size, proposals_num, rs_filter_num);
    cfa_model = tf.keras.Model(inputs=input_image, outputs=output);         # The callback object for the learning rate
    callback_lr = tf.keras.callbacks.LearningRateScheduler(scheduler);
    cfa_callbacks = [callback_lr];
    return [cfa_model, cfa_callbacks];

# Probability Mask Filter + Henz's demosaicer
def ProbMask_Henz(imheight, imwidth, channel, filter_size, pmask_slope, sample_slope, sparsity):
    input_image = tf.keras.Input(shape=(imheight, imwidth, channel));
    filtered_image = cfa.ProbMaskFilter(pmask_slope, sample_slope, filter_size, channel, sparsity)(input_image);
    reduced_image = tf.reduce_sum(filtered_image, axis=-1, keepdims=True);
    output = dm.demosaicer_henz(reduced_image, filtered_image);
    cfa_model = tf.keras.Model(inputs=input_image, outputs=output);
    callback_lr = tf.keras.callbacks.LearningRateScheduler(scheduler);      # The callback object for the learning rate
    cfa_callbacks = [callback_lr];
    return [cfa_model, cfa_callbacks];

# Probability Mask Filter + de Gioia's demosaicer
def ProbMask_deGioia(imheight, imwidth, channel, filter_size, pmask_slope, sample_slope, sparsity, demos_filter_num):
    input_image = tf.keras.Input(shape=(imheight, imwidth, channel));
    filtered_image = cfa.ProbMaskFilter(pmask_slope, sample_slope, filter_size, channel, sparsity)(input_image);
    output = dm.demosaicer_degiogia(filtered_image, demos_filter_num);
    cfa_model = tf.keras.Model(inputs=input_image, outputs=output);
    callback_lr = tf.keras.callbacks.LearningRateScheduler(scheduler);      # The callback object for the learning rate
    cfa_callbacks = [callback_lr];
    return [cfa_model, cfa_callbacks];

# Probability Mask Filter + Custom demosaicer 
def ProbMask_Custom(imheight, imwidth, channel, filter_size, pmask_slope, sample_slope, sparsity):
    input_image = tf.keras.Input(shape=(imheight, imwidth, channel));
    filtered_image = cfa.ProbMaskFilter(pmask_slope, sample_slope, filter_size, channel, sparsity)(input_image);
    [proxy, inter_1, inter_2, inter_3, output_image]= dm.demosaicer_custom(filtered_image);
    cfa_model = tf.keras.Model(inputs=input_image, outputs=output_image);
    callback_lr = tf.keras.callbacks.LearningRateScheduler(scheduler);      # The callback object for the learning rate
    #callback_save = SaveEpochs()
    cfa_callbacks = [callback_lr];
    #cfa_callbacks = [callback_lr,
    #                 callback_save]
    return [cfa_model, cfa_callbacks];

""" Joint Models with Maximum Thresholding Filter"""
# Maximum Thresholding Filter + Chakrabarti's demosaicer
def MaxThreshold_Chak(imheight, imwidth, channel, filter_size, proposals_num, rs_filter_num):
    input_image = tf.keras.Input(shape=(imheight, imwidth, channel));
    filtered_image = cfa.MaxThresholdFilter(filter_size = filter_size, channels = channel)(input_image);
    filtered_image_merged = tf.reduce_sum(filtered_image, axis=-1, keepdims=True);
    output = dm.demosaicer_chak(filtered_image_merged, filter_size, proposals_num, rs_filter_num);
    cfa_model = tf.keras.Model(inputs=input_image, outputs=output);         # The callback object for the learning rate
    callback_lr = tf.keras.callbacks.LearningRateScheduler(scheduler);
    cfa_callbacks = [callback_lr];
    return [cfa_model, cfa_callbacks];

# Maximum Thresholding Filter + Henz's demosaicer
def MaxThreshold_Henz(imheight, imwidth, channel, filter_size):
    input_image = tf.keras.Input(shape=(imheight, imwidth, channel));
    filtered_image = cfa.MaxThresholdFilter(filter_size = filter_size, channels = channel)(input_image);
    reduced_image = tf.reduce_sum(filtered_image, axis=-1, keepdims=True);
    output = dm.demosaicer_henz(reduced_image, filtered_image);
    cfa_model = tf.keras.Model(inputs=input_image, outputs=output);
    callback_lr = tf.keras.callbacks.LearningRateScheduler(scheduler);      # The callback object for the learning rate
    cfa_callbacks = [callback_lr];
    return [cfa_model, cfa_callbacks];

# Maximum Thresholding Filter + de Gioia's demosaicer
def MaxThreshold_deGioia(imheight, imwidth, channel, filter_size, demos_filter_num):
    input_image = tf.keras.Input(shape=(imheight, imwidth, channel));
    filtered_image = cfa.MaxThresholdFilter(filter_size = filter_size, channels = channel)(input_image);
    output = dm.demosaicer_degiogia(filtered_image, demos_filter_num);
    cfa_model = tf.keras.Model(inputs=input_image, outputs=output);
    callback_lr = tf.keras.callbacks.LearningRateScheduler(scheduler);      # The callback object for the learning rate
    cfa_callbacks = [callback_lr];
    return [cfa_model, cfa_callbacks];

# Maximum Thresholding Filter + Custom demosaicer 
def MaxThreshold_Custom(imheight, imwidth, channel, filter_size):
    input_image = tf.keras.Input(shape=(imheight, imwidth, channel));
    filtered_image = cfa.MaxThresholdFilter(filter_size = filter_size, channels = channel)(input_image);
    [proxy, inter_1, inter_2, inter_3, output_image]= dm.demosaicer_custom(filtered_image);
    cfa_model = tf.keras.Model(inputs=input_image, outputs=output_image);
    callback_lr = tf.keras.callbacks.LearningRateScheduler(scheduler);      # The callback object for the learning rate
    #callback_save = SaveEpochs()
    cfa_callbacks = [callback_lr];
    #cfa_callbacks = [callback_lr,
    #                 callback_save]
    return [cfa_model, cfa_callbacks];

"""
CFA Reconstruction model working with images that are already filtered, such
as Bayer. The output is a BGR image.
 
    Input Masked Image Shape:   3P x 3P x 1
    Multiplicative shape:       P x P x 3K
    Convolution shape:          P x P x 3K
    Output shape:               P x P x 3
"""
def FixedFilter_Chak(imheight, imwidth, sm_filter_size, proposals_num, rs_filter_num):
    input_image = tf.keras.Input(shape=(imheight, imwidth, 1))
    output = dm.demosaicer_chak(input_image, sm_filter_size, proposals_num, rs_filter_num)
    cfa_model = tf.keras.Model(inputs=input_image, outputs=output)
    # The callback object for the learning rate
    callback_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
    cfa_callbacks = [callback_lr]
    return [cfa_model, cfa_callbacks]

def FixedFilter_Henz(imheight, imwidth):
    input_image = tf.keras.Input(shape=(imheight, imwidth, 1))
    output = dm.demosaicer_henz(input_image)
    cfa_model = tf.keras.Model(inputs=input_image, outputs=output)
    # The callback object for the learning rate
    callback_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
    cfa_callbacks = [callback_lr]
    return [cfa_model, cfa_callbacks]

def FixedFilter_deGioia(imheight, imwidth, demos_filter_num):
    input_image = tf.keras.Input(shape=(imheight, imwidth, 1));
    output = dm.demosaicer_degiogia(input_image, demos_filter_num);
    cfa_model = tf.keras.Model(inputs=input_image, outputs=output);
    callback_lr = tf.keras.callbacks.LearningRateScheduler(scheduler);      # The callback object for the learning rate
    cfa_callbacks = [callback_lr];
    return [cfa_model, cfa_callbacks];