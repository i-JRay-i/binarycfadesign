import tensorflow as tf
import numpy as np
from os.path import exists
from os import makedirs
from math import ceil
import joint_models as cfa
import cfa_models as fil

# Model Hyperparameters
CFA_FILTER = "MaxThreshold"
DEMOSAICER = "Custom"
NORMALIZE = False

P = 8       # Filter Size
C = 4       # Number of channel
K = 3*P     # Number of proposals
F = 128     # Reconstruction network filter size

IMAGE_ADDRESS = "..\\dataset\\training_blocks_" + str(P) + "_large.npy"   # Initializing the address
#IMAGE_ADDRESS_2 = "..\\dataset\\training_blocks_8_large.npy"   # Initializing the address
FILTER_ADDRESS = "..\\filters\\max_threshold_chak_bgr_8_filter.npy"
IMAGE_HEIGHT = 3*P
IMAGE_WIDTH = 3*P

# Training parameters
BATCH_SIZE = 128
EPOCH = 100
LR = 0.001

# Loading and preparing the training dataset
if C == 3:
    training_input = np.load(IMAGE_ADDRESS)[:1000000,:,:,1:]
    training_input_num = training_input.shape[0]
    training_output = training_input
    if DEMOSAICER == "Chakrabarti":
        training_output = training_output[:,P:2*P,P:2*P]
elif C == 4:
    training_input = np.load(IMAGE_ADDRESS)[:1000000,:,:,:]
    training_input_num = training_input.shape[0]
    training_output = training_input[:,:,:,1:]
    if DEMOSAICER == "Chakrabarti":
        training_output = training_output[:,P:2*P,P:2*P]

# Normalizing the dataset
if NORMALIZE: 
    training_input = (training_input / 255).astype("float32")
    training_output = (training_output / 255).astype("float32")

print("Input Shape: ", training_input.shape)
print("Output Shape: ", training_output.shape)

# Initializing the model
if CFA_FILTER == "WeightedSoftmax":
    INITIAL_ALPHA = 1
    GAMMA = 2.5*10**(-5)
    if DEMOSAICER == "Henz": 
        [model, callbacks] = cfa.WeightedSoftmax_Henz(IMAGE_HEIGHT, IMAGE_WIDTH, 
                                                      channel=C, sm_filter_size=P, gamma=GAMMA,
                                                      batch_size=BATCH_SIZE, input_num=training_input_num)
    elif DEMOSAICER == "Chakrabarti":
        [model, callbacks] = cfa.WeightedSoftmax_Chak(IMAGE_HEIGHT, IMAGE_WIDTH, 
                                                      channel=C, sm_filter_size=P, gamma=GAMMA,
                                                      proposals_num=K, rs_filter_num=F,
                                                      batch_size=BATCH_SIZE, input_num=training_input_num)
    elif DEMOSAICER == "deGioia":
        [model, callbacks] = cfa.WeightedSoftmax_deGioia(IMAGE_HEIGHT, IMAGE_WIDTH, 
                                                      channel=C, sm_filter_size=P, demos_filter_num=64,
                                                      gamma=GAMMA, batch_size=BATCH_SIZE, input_num=training_input_num)
    elif DEMOSAICER == "Custom":
        [model, callbacks] = cfa.WeightedSoftmax_Custom(IMAGE_HEIGHT, IMAGE_WIDTH, 
                                                      channel=C, sm_filter_size=P, gamma=GAMMA,
                                                      batch_size=BATCH_SIZE, input_num=training_input_num)
elif CFA_FILTER == "Linear":
    if DEMOSAICER == "Henz": 
        [model, callbacks] = cfa.Linear_Henz(IMAGE_HEIGHT, IMAGE_WIDTH, 
                                                   channel=C, filter_size=P)
    elif DEMOSAICER == "Chakrabarti":
        [model, callbacks] = cfa.Linear_Chak(IMAGE_HEIGHT, IMAGE_WIDTH, 
                                                   channel=C, filter_size=P,
                                                   proposals_num=K, rs_filter_num=F)    
    elif DEMOSAICER == "deGioia":
        [model, callbacks] = cfa.Linear_deGioia(IMAGE_HEIGHT, IMAGE_WIDTH, 
                                                      channel=C, filter_size=P,
                                                      demos_filter_num = 64)
    elif DEMOSAICER == "Custom":
        [model, callbacks] = cfa.Linear_Custom(IMAGE_HEIGHT, IMAGE_WIDTH, 
                                                      channel=C, filter_size=P)

elif CFA_FILTER == "MaxThreshold":
    if DEMOSAICER == "Henz": 
        [model, callbacks] = cfa.MaxThreshold_Henz(IMAGE_HEIGHT, IMAGE_WIDTH, 
                                                   channel=C, filter_size=P)
    elif DEMOSAICER == "Chakrabarti":
        [model, callbacks] = cfa.MaxThreshold_Chak(IMAGE_HEIGHT, IMAGE_WIDTH, 
                                                   channel=C, filter_size=P,
                                                   proposals_num=K, rs_filter_num=F)
    elif DEMOSAICER == "deGioia":
        [model, callbacks] = cfa.MaxThreshold_deGioia(IMAGE_HEIGHT, IMAGE_WIDTH, 
                                                      channel=C, filter_size=P,
                                                      demos_filter_num = 64)
    elif DEMOSAICER == "Custom":
        [model, callbacks] = cfa.MaxThreshold_Custom(IMAGE_HEIGHT, IMAGE_WIDTH, 
                                                      channel=C, filter_size=P)

elif CFA_FILTER == "Bayer":
    bayer_filter = fil.bayer_filter(3*P)
    filtered_input = []
    for img in training_input:
        img_filtered = np.multiply(img, bayer_filter)
        img_mask = np.sum(img_filtered, axis=2)
        img_mask = np.expand_dims(img_mask, axis=2)
        filtered_input.append(img_mask)
    training_input = np.asarray(filtered_input)
    if DEMOSAICER == "Henz": 
        [model, callbacks] = cfa.FixedFilter_Henz(IMAGE_HEIGHT, IMAGE_WIDTH)
    elif DEMOSAICER == "Chakrabarti":
        [model, callbacks] = cfa.FixedFilter_Chak(IMAGE_HEIGHT, IMAGE_WIDTH, 
                                              sm_filter_size=P, 
                                              proposals_num=K, rs_filter_num=F)
elif CFA_FILTER == "CFA":
    cfz_filter = fil.cfz_filter(3*P)
    filtered_input = []
    for img in training_input:
        img_filtered = np.multiply(img, cfz_filter)
        img_mask = np.sum(img_filtered, axis=2)
        img_mask = np.expand_dims(img_mask, axis=2)
        filtered_input.append(img_mask)
    training_input = np.asarray(filtered_input)
    if DEMOSAICER == "Henz": 
        [model, callbacks] = cfa.FixedFilter_Henz(IMAGE_HEIGHT, IMAGE_WIDTH)
    elif DEMOSAICER == "Chakrabarti":
        [model, callbacks] = cfa.FixedFilter_Chak(IMAGE_HEIGHT, IMAGE_WIDTH, 
                                              sm_filter_size=P, 
                                              proposals_num=K, rs_filter_num=F)
elif CFA_FILTER == "Custom":
    custom_filter = fil.custom_filter(FILTER_ADDRESS)
    filtered_input = []
    for img in training_input:
        img_filtered = np.multiply(img, custom_filter)
        img_mask = np.sum(img_filtered, axis=2)
        img_mask = np.expand_dims(img_mask, axis=2)
        filtered_input.append(img_mask)
    training_input = np.asarray(filtered_input)
    if DEMOSAICER == "Henz": 
        [model, callbacks] = cfa.FixedFilter_Henz(IMAGE_HEIGHT, IMAGE_WIDTH)
    elif DEMOSAICER == "Chakrabarti":
        [model, callbacks] = cfa.FixedFilter_Chak(IMAGE_HEIGHT, IMAGE_WIDTH, 
                                              sm_filter_size=P, 
                                              proposals_num=K, rs_filter_num=F)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=["mae"])
#model.summary()

history = model.fit(x=training_input,
                    y=training_output,
                    epochs=EPOCH,
                    batch_size=BATCH_SIZE,
                    steps_per_epoch=ceil(training_input_num/BATCH_SIZE),
                    callbacks=callbacks)

if not exists(".\\models"):
    makedirs(".\\models")

MODEL_NAME = ".\\models\\cfa_demosaicer"
if CFA_FILTER == "MaxThreshold":
    MODEL_NAME = MODEL_NAME + "_max_threshold"
elif CFA_FILTER == "WeightedSoftmax":
    MODEL_NAME = MODEL_NAME + "_weighted_softmax"
elif CFA_FILTER == "Custom":
    MODEL_NAME = MODEL_NAME + "_fixed"
elif CFA_FILTER == "Linear":
    MODEL_NAME = MODEL_NAME + "_linear"
if DEMOSAICER == "Henz":
    MODEL_NAME = MODEL_NAME + "_henz"
elif DEMOSAICER == "Chakrabarti":
    MODEL_NAME = MODEL_NAME + "_chak"
elif DEMOSAICER == "deGioia":
    MODEL_NAME = MODEL_NAME + "_degioia"
if C == 3:
    MODEL_NAME = MODEL_NAME + "_bgr"
elif C == 4:
    MODEL_NAME = MODEL_NAME + "_cbgr"
#MODEL_NAME = MODEL_NAME + "_" + str(P)
#MODEL_NAME = MODEL_NAME + "_1mil"
model.save(MODEL_NAME)

print(history.history)