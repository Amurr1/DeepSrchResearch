import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import tensorflow as tf
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('float32')
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization as BatchNorm
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from tensorflow.keras import metrics, layers, Model, Input
from keras_cv_attention_models import efficientformer
import tensorflow_addons as tfa
import numpy as np
from datetime import date
import argparse
import time

def get_deluxe():
    path = '/global/cfs/projectdirs/cosmo/work/users/xhuang/dr10_1/Clean-Samples/TS40_deluxe_clean'
    
    xtrain = np.load(f"{path}/train_x.npy")
    ytrain = np.load(f"{path}/train_y.npy").reshape(-1, 1)
    xval = np.load(f"{path}/val_x.npy")
    yval = np.load(f"{path}/val_y.npy").reshape(-1, 1)
    
    xtrain = np.clip(xtrain, -1, 1)  
    xval = np.clip(xval, -1, 1)

    return xtrain, ytrain, xval, yval

def get_ethan():
    path = '/global/cfs/projectdirs/deepsrch/jwst_sims/pristine_bright/'
    
    x0 = np.load(path+"images.npy")
    y0 = np.load(path+"lensed.npy")
    
    cap = np.percentile(x0, 99)
    for i in range(len(x0)):
        cap = np.percentile(x0[i],99)
        x0[i][x0[i]>cap]=cap
        x0[i] = (x0[i]-np.mean(x0[i])) / np.std(x0[i])
    
    np.random.seed(15)
    indices = np.arange(len(x0))
    np.random.shuffle(indices)
    start = len(x0)//5 * 0 #0
    end = len(x0)//5 * 1 #1
    val_inds = indices[start:end]
    train_inds = np.concatenate([indices[:start],indices[end:]])
    
    xtrain = x0[train_inds]
    xval = x0[val_inds]
    ytrain = y0[train_inds]
    yval = y0[val_inds]
    
    xtrain = np.reshape(xtrain,(len(xtrain),125,125,1))
    xval =  np.reshape(xval, (len(xval),125,125,1))
    ytrain = np.reshape(ytrain, (len(ytrain),1))
    yval =  np.reshape(yval, (len(yval),1))
    
    xtrain = np.clip(xtrain, -1, 1)  
    xval = np.clip(xval, -1, 1)
    if xtrain.ndim == 3:
        xtrain = np.expand_dims(xtrain, axis=-1)
    if xval.ndim == 3:
        xval = np.expand_dims(xval, axis=-1)
        
    return xtrain, ytrain, xval, yval
    
def ensure_rgb(x):
    if x.shape.rank == 3 and x.shape[-1] == 1:
        x = tf.image.grayscale_to_rgb(x)
    return x
    
def preprocess_image(x, y):
    x = ensure_rgb(x)
    x = tf.image.random_flip_left_right(tf.image.random_flip_up_down(x))
    rg = tf.random.uniform(shape=[],minval=0, maxval=2 * np.pi, dtype=tf.float32)
    x = tfa.image.rotate(x, angles=rg, fill_mode = 'reflect')
    return x, y
    
def preprocess_image_val(x, y):
    x = ensure_rgb(x)
    return x, y
    
def preprocess_dataset(ds="deluxe", bs=1024):
    xtrain = None
    ytrain = None
    xval = None 
    yval = None 
    input_shape = None
    if ds == "deluxe":
        xtrain, ytrain, xval, yval = get_deluxe()
        input_shape = (101,101,3)
    if ds == "ethan":
        input_shape = (125,125,3)
        xtrain, ytrain, xval, yval = get_ethan()

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    
    train = (tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            .map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .shuffle(len(ytrain), reshuffle_each_iteration=True, seed=42) 
            .repeat()
            .batch(bs)
            .prefetch(tf.data.experimental.AUTOTUNE)).with_options(options)
    validate = (tf.data.Dataset.from_tensor_slices((xval, yval))
            .map(preprocess_image_val, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .shuffle(len(yval))
            .repeat()
            .batch(bs)
            .prefetch(tf.data.experimental.AUTOTUNE)).with_options(options)

    steps_per_epoch = (len(ytrain) // bs)
    validation_steps = (len(yval) // bs)

    return train, validate, input_shape, steps_per_epoch, validation_steps

def create_eff_former(model_name, lr, input_shape):
    base_model = None
    if model_name == "s0":
        base_model = efficientformer.EfficientFormerV2S0(input_shape=input_shape, pretrained='imagenet')
    elif model_name == "s1":
        base_model = efficientformer.EfficientFormerV2S1(input_shape=input_shape, pretrained='imagenet')
    elif model_name == "s2":
        base_model = efficientformer.EfficientFormerV2S2(input_shape=input_shape, pretrained='imagenet')
    elif model_name == "l":
        base_model = efficientformer.EfficientFormerV2L(input_shape=input_shape, pretrained='imagenet')
    
    base_model.trainable = True
    
    x = base_model.layers[-2].output
    x = Dense(64, activation='relu')(x)
    output = Dense(1, activation='sigmoid', dtype='float32')(x)

    model = tf.keras.models.Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = lr),
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics = [
            metrics.AUC(num_thresholds=1000), 
            metrics.Precision(0.9), 
            metrics.Recall(0.9),
        ],
    )
    return model

def train_model(model_name, train, validate, input_shape, steps_per_epoch, validation_steps, lr, bs, log_csv, w_chkpt, w_end):
    model = create_eff_former(model_name, lr, input_shape)
    
    start = time.time()
    print(f'Start: {start}')

    checkpoint = ModelCheckpoint(
        w_chkpt, 
        monitor = f'val_auc', 
        save_best_only = True, 
        mode = 'max', 
        verbose = 1, 
        save_weights_only = True,
    )
    
    csv_logger = CSVLogger(
        log_csv, 
        separator = ',', 
        append = True,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1,
        min_lr=1e-7
    )
    
    callbacks = [
        checkpoint, 
        csv_logger,
        reduce_lr,
    ]
    
    model.fit(
        train, 
        validation_data = validate, 
        epochs = 1, 
        steps_per_epoch = steps_per_epoch, 
        callbacks = callbacks, 
        verbose = 1, 
        batch_size = bs, 
        validation_steps = validation_steps,
    )
    
    end = time.time()
    print(f'Total time running: {end-start}')
    model.save_weights(w_end)

def main():
    parser = argparse.ArgumentParser(description="Train model with given hyperparameters and dataset")
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--log_csv', type=str, default=None, help='Path to CSV for logging')
    parser.add_argument('--w_chkpt', type=str, default=None, help='Path to best weights')
    parser.add_argument('--w_end', type=str, default=None, help='Path to end weights')
    args = parser.parse_args()
    model_name = args.model
    dataset = "deluxe"
    learning_rate = 0.0005
    batch_size = 512
    log_csv = args.log_csv
    w_chkpt = args.w_chkpt
    w_end = args.w_end

    print(f"Training on dataset: {dataset}")
    print(f"Using learning rate: {learning_rate}")
    print(f"Using batch size: {batch_size}")

    tf.config.list_physical_devices("GPU")
    train, validate, input_shape, steps_per_epoch, validation_steps = preprocess_dataset(ds=dataset, bs=batch_size)
    train_model(
        model_name, 
        train, 
        validate, 
        input_shape, 
        steps_per_epoch, 
        validation_steps, 
        learning_rate, 
        batch_size, 
        log_csv, 
        w_chkpt, 
        w_end
    )

if __name__ == "__main__":
    main()