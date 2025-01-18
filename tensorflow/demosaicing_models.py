"""
This Python file contains a variety of demosaicing methods adapted 
from multiple joint filtering / demosaicing architectures and a unique 
demosaicer.

Written by: Cemre Ömer Ayna

References: 
"""
import tensorflow as tf

#The demosaicing model proposed in Chakrabarti
"""
    input = 3*P x 3*P x 1
    x   = P * P * 3*K x 1
    x   = P * P * 3*K x 1
    x   = P * P * 3*K x 1
    x   = P x P x 3*K
    x   = P x P x 3*K
"""
def reconstruction_stream_1(input_image, sm_filter_size, proposals_num):
    x = tf.keras.layers.Flatten()(input_image)
    x = tf.cast(x, dtype=tf.float32)
    #x = tf.keras.backend.log(x + 0.1) 
    x = tf.keras.layers.Dense(units=sm_filter_size*sm_filter_size*3*proposals_num, 
                              use_bias=False,
                              kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001))(x)
    #x = tf.keras.backend.exp(x)
    x = tf.keras.layers.Reshape(target_shape=(sm_filter_size,sm_filter_size,3*proposals_num))(x)
    output_image_x = tf.keras.layers.Conv2D(filters=3*proposals_num, kernel_size=(1,1), strides=(1,1))(x)
    return output_image_x

"""
    input = 3*P x 3*P x 1
    x     = 3 x 3 x F
    x     = 2 x 2 x F
    x     = 1 x 1 x F
    x     = 8 x 8 x 3*K
"""
def reconstruction_stream_2(input_image, sm_filter_size, proposals_num, filter_num):
    y = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(sm_filter_size,sm_filter_size), 
                               strides=(sm_filter_size,sm_filter_size))(input_image)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(2,2))(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(2,2))(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.Flatten()(y)
    y = tf.keras.layers.Dense(units=sm_filter_size*sm_filter_size*3*proposals_num)(y)
    y = tf.keras.layers.ReLU()(y)
    output_image_y = tf.keras.layers.Reshape(target_shape=(sm_filter_size,sm_filter_size,3*proposals_num))(y)
    return output_image_y

def demosaicer_chak(filtered_image, sm_filter_size, proposals_num, rs_filter_num):
    stream1 = reconstruction_stream_1(filtered_image, sm_filter_size, proposals_num)
    stream2 = reconstruction_stream_2(filtered_image, sm_filter_size, proposals_num, rs_filter_num)

    stream1_b = tf.keras.layers.Lambda(lambda x: x[:,:,:,:proposals_num])(stream1)
    stream1_g = tf.keras.layers.Lambda(lambda x: x[:,:,:,proposals_num:2*proposals_num])(stream1)
    stream1_r = tf.keras.layers.Lambda(lambda x: x[:,:,:,2*proposals_num:])(stream1)
    
    stream2_b = tf.keras.layers.Lambda(lambda x: x[:,:,:,:proposals_num])(stream2)
    stream2_g = tf.keras.layers.Lambda(lambda x: x[:,:,:,proposals_num:2*proposals_num])(stream2)
    stream2_r = tf.keras.layers.Lambda(lambda x: x[:,:,:,2*proposals_num:])(stream2)
    
    output_b = tf.math.reduce_sum(tf.keras.layers.Multiply()([stream1_b , stream2_b]), axis=-1, keepdims=True)
    output_g = tf.math.reduce_sum(tf.keras.layers.Multiply()([stream1_g , stream2_g]), axis=-1, keepdims=True)
    output_r = tf.math.reduce_sum(tf.keras.layers.Multiply()([stream1_r , stream2_r]), axis=-1, keepdims=True)
    
    output = tf.keras.layers.Concatenate()([output_b, output_g, output_r])
    return output

#The demosaicing model proposed in Henz
"""
    submosaics              = 3P x 3P x 3
    interpolated submosaics = 3P x 3P x 3
    mosaic                  = 3P x 3P x 1
"""

def henz_input(mosaic, submosaic, filter_size):
    k = tf.expand_dims(tf.range(1, filter_size, 1),0)
    k = tf.concat([k, [[filter_size]], tf.reverse(k, [1])], 1)
    k = tf.cast(tf.divide(k,filter_size),tf.float32)
    k = tf.matmul(tf.transpose(k),k)[:,:,tf.newaxis,tf.newaxis]
    
    submosaic_b = tf.expand_dims(submosaic[:,:,:,0],3)
    submosaic_g = tf.expand_dims(submosaic[:,:,:,1],3)
    submosaic_r = tf.expand_dims(submosaic[:,:,:,2],3)

    intrpl_b = tf.nn.conv2d(submosaic_b, k, strides=1, padding='SAME')
    intrpl_g = tf.nn.conv2d(submosaic_g, k, strides=1, padding='SAME')
    intrpl_r = tf.nn.conv2d(submosaic_r, k, strides=1, padding='SAME')
    
    # mosaic = tf.reduce_sum(submosaic, axis=-1, keepdims=True)
    
    intrpl = tf.concat([intrpl_b, intrpl_g, intrpl_r], 3)
    input_data = tf.concat([mosaic, submosaic, intrpl], 3)
    return input_data

def conv_block(input_image, filter_num):
    x = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3,3), padding="same")(input_image)
    x = tf.keras.layers.BatchNormalization()(x)
    output_image = tf.keras.layers.ReLU()(x)
    return output_image

def demosaicer_henz(filtered_image, weighted_image):
    input_data = henz_input(filtered_image, weighted_image, 8)
    conv1 = conv_block(input_data, 64)
    conv2 = conv_block(conv1, 64)
    conv3 = conv_block(conv2, 64)
    conv4 = conv_block(conv3, 64)
    conv5 = conv_block(conv4, 64)
    conv6 = conv_block(conv5, 64)
    conv7 = conv_block(conv6, 128)
    conv8 = conv_block(conv7, 128)
    conv9 = conv_block(conv8, 128)
    conv10 = conv_block(conv9, 128)
    conv11 = conv_block(conv10, 128)
    conv12 = conv_block(conv11, 128)
    concat = tf.keras.layers.Concatenate()([filtered_image, conv12])
    x = tf.keras.layers.Conv2D(filters=3, kernel_size=(3,3), padding="same")(concat)
    output_image = tf.keras.layers.ReLU()(x)
    return output_image

#The demosaicing model proposed in de Gioia
def luminance_stream(input_image, filter_size, filter_num):
    y = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(filter_size,filter_size), padding="same")(input_image)
    y = tf.keras.activations.sigmoid(y)
    luminance = tf.keras.layers.Conv2D(filters=1, kernel_size=(filter_size,filter_size), padding="same")(y)
    return luminance

def chrominance_stream(input_image, filter_size, filter_num):
    y = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(filter_size,filter_size), 
                               strides=(2,2), padding="same")(input_image)
    y = tf.keras.activations.relu(y)
    y = tf.keras.layers.Conv2D(filters=2*filter_num, kernel_size=(filter_size,filter_size), 
                               strides=(2,2), padding="same")(y)
    y = tf.keras.activations.relu(y)
    y = tf.keras.layers.Conv2DTranspose(filters=filter_num, kernel_size=(filter_size,filter_size),
                                        strides=(1, 1), dilation_rate=(3, 3))(y)
    y = tf.keras.activations.relu(y)
    chrominance = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(filter_size,filter_size),
                                        strides=(1, 1), dilation_rate=(6, 6))(y)
    return chrominance

def demosaicer_degiogia(filtered_image, filter_num):
    luminance = luminance_stream(filtered_image, 3, filter_num)
    chrominance = chrominance_stream(filtered_image, 3, filter_num)
    output_image = tf.keras.layers.Add()([luminance, chrominance])
    return output_image
    
# Custom Demosaicing Model
def custom_conv_block(input_image, filter_num, filter_size):
    x = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(filter_size, filter_size),
                               padding="same", kernel_regularizer=tf.keras.regularizers.l2(5e-4))(input_image)
    output_image = tf.keras.layers.ReLU()(x)
    return output_image
    
def demosaicer_custom(filtered_image):
    pseudoimage = tf.keras.layers.Conv2D(filters=3, kernel_size=(9, 9), padding="same", use_bias=False)(filtered_image)
    relu = tf.keras.layers.ReLU()(pseudoimage)
    conv1_1 = custom_conv_block(relu, 64, 3)
    conv1_2 = custom_conv_block(conv1_1, 32, 3)
    conv1_3 = custom_conv_block(conv1_2, 3, 3)
    concat_1 = tf.keras.layers.Concatenate()([pseudoimage, conv1_3])
    conv2_1 = custom_conv_block(concat_1, 64, 3)
    conv2_2 = custom_conv_block(conv2_1, 32, 3)
    conv2_3 = custom_conv_block(conv2_2, 3, 3)
    concat_2 = tf.keras.layers.Concatenate()([conv1_3, conv2_3])
    conv3_1 = custom_conv_block(concat_2, 64, 3)
    conv3_2 = custom_conv_block(conv3_1, 32, 3)
    conv3_3 = custom_conv_block(conv3_2, 3, 3)
    concat_3 = tf.keras.layers.Concatenate()([conv2_3, conv3_3])
    output_image = custom_conv_block(concat_3, 3, 3)
    return [pseudoimage, conv1_3, conv2_3, conv3_3, output_image]