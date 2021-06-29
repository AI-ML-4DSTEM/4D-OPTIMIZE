import sys
import h5py
import numpy as np
from tqdm import tqdm
import tensorflow as tf

#Custom libraries - User defined
import convSpectral
import poolSpectral
import augmentation

'''
These are custom Loss functions
'''

class CustomL1Loss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        
        return tf.math.reduce_mean(tf.math.abs(y_true - y_pred), axis=-1)

class customSSIMLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        
        data_range = tf.math.reduce_max(y_true) - tf.math.reduce_min(y_true)
        ssim_loss = 1-tf.image.ssim(y_true, y_pred, max_val=data_range)
        
        return ssim_loss
    
class CustomSSIML1ELoss(tf.keras.losses.Loss):    
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        
        data_range = tf.math.reduce_max(y_true) - tf.math.reduce_min(y_true)
        ssim_loss = 1-tf.image.ssim(y_true, y_pred, max_val=data_range)
        mse_loss = tf.math.reduce_mean(tf.math.abs(y_true - y_pred), axis=(1,2,3))
        
        loss = ssim_loss + mse_loss
        
        return loss

'''
This class is to load tensorflow records from the database
'''
# %%
# Read TFRecord file
class parseDataset(object):
    def __init__(self, filepath = ''):
        self.ext = filepath.split('.')[-1]
        assert(self.ext == 'hdf5' or self.ext == 'tfrecords'), "Currently only supports hdf5 or tfrecords as dataset" 
        self.filepath = filepath
        
    def read(self, batch_size = 128, shuffle = True, augment = True, mode = 'default'):
        if self.ext == 'hdf5':
            ds = self.prepare_dataset_from_hdf5(batch_size , shuffle , augment, mode)
        elif self.ext == 'tfrecords':
            ds = self.prepare_dataset_from_tfrecords(batch_size , shuffle , augment, mode)
        return ds
            
    def prepare_dataset_from_tfrecords(self, batch_size , shuffle , augment, mode):
        '''
        parse data from tfrecords
        '''
        #TODO: Scaling scheme and standardization
        tfr_dataset = tf.data.TFRecordDataset(self.filepath) 
    
        if mode == 'default':
            ds = tfr_dataset.map(self._parse_tfr_element)
        elif mode == 'norm':
            ds = tfr_dataset.map(self._parse_tfr_element_norm)
        elif mode == 'multitask':
            ds = tfr_dataset.map(self._parse_tfr_element_multitask)
    
        if shuffle:
            SHUFFLE_BUFFER_SIZE = 10000
            ds = ds.shuffle(SHUFFLE_BUFFER_SIZE)
        
        ds = ds.batch(batch_size, drop_remainder = True)
    
        if augment:
            data_augment = tf.keras.layers.Lambda(self.data_augmentation)
            ds_aug = ds.map(lambda x, y: (data_augment(x, training = True), y))
            ds = ds.concatenate(ds_aug)
        
        return ds

    def prepare_dataset_from_hdf5(self, batch_size , shuffle , augment, mode):
        '''
        parse data from hdf5
        '''
        data = h5py.File(self.filepath,'r')

        cbed = np.array(data.get('cbed').get('cbed_data'))
        probe = np.array(data.get('probe').get('probe_data'))
    
        if mode == 'default':
            pot = np.array(data.get('pot').get('pot_data'))
            ds = tf.data.Dataset.from_tensor_slices(((cbed,probe), pot))
        elif mode == 'norm':
            pot = np.array(data.get('pot').get('pot_data'))
            shape = pot.shape[0]
            pot = pot / np.amax(test_probe, axis = (1,2)).reshape(shape)
            ds = tf.data.Dataset.from_tensor_slices(((cbed,probe), pot))
        elif mode == 'multitask':
            pot = np.array(data.get('pot').get('pot_data'))
            shape = pot.shape[0]
            pot_max = np.amax(test_probe, axis = (1,2)).reshape(shape)
            pot = pot / pot_max
            ds = tf.data.Dataset.from_tensor_slices(((cbed,probe), (pot, pot_max)))
    
        if shuffle:
            SHUFFLE_BUFFER_SIZE = 10000
            ds = ds.shuffle(SHUFFLE_BUFFER_SIZE)
        
        ds = ds.batch(batch_size)
    
        if augment:
            data_augment = tf.keras.layers.Lambda(self.data_augmentation)
            ds_aug = ds.map(lambda x, y: (data_augment(x, training = True), y))
            ds = ds.concatenate(ds_aug)
        
        return ds

    def _parse_tfr_element(self, element):
        parse_dic = {
            'cbed_feature': tf.io.FixedLenFeature([], tf.string), 
            'probe_feature': tf.io.FixedLenFeature([], tf.string),
            'pot_feature': tf.io.FixedLenFeature([], tf.string),
            }
        example_message = tf.io.parse_single_example(element, parse_dic)
        paddings = tf.constant([[3, 3,], [3, 3], [0,0]])

        cbed_feature = example_message['cbed_feature'] 
        probe_feature = example_message['probe_feature']
        pot_feature = example_message['pot_feature']
        cbed = tf.io.parse_tensor(cbed_feature, out_type=tf.float32)
        cbed.set_shape([250, 250, 1])
        cbed = tf.image.resize(cbed, (256,256))
        probe = tf.io.parse_tensor(probe_feature, out_type=tf.float32)
        probe.set_shape([250, 250, 1])
        probe = tf.image.resize(probe, (256,256))
        pot = tf.io.parse_tensor(pot_feature, out_type=tf.float32)
        pot.set_shape([250, 250, 1])
        pot = tf.image.resize(pot, (256,256))
        return ((cbed,probe),pot)

    def _parse_tfr_element_norm(self, element):
        parse_dic = {
            'cbed_feature': tf.io.FixedLenFeature([], tf.string), 
            'probe_feature': tf.io.FixedLenFeature([], tf.string),
            'pot_feature': tf.io.FixedLenFeature([], tf.string),
            }
        example_message = tf.io.parse_single_example(element, parse_dic)

        cbed_feature = example_message['cbed_feature'] 
        probe_feature = example_message['probe_feature']
        pot_feature = example_message['pot_feature']
        cbed = tf.io.parse_tensor(cbed_feature, out_type=tf.float32)
        cbed.set_shape([250, 250, 1])
        cbed = tf.image.resize(cbed, (256,256))
        probe = tf.io.parse_tensor(probe_feature, out_type=tf.float32)
        probe.set_shape([250, 250, 1])
        probe = tf.image.resize(probe, (256,256))
        pot = tf.io.parse_tensor(pot_feature, out_type=tf.float32)
        pot.set_shape([250, 250, 1])
        pot = tf.image.resize(pot, (256,256))
        pot = pot/tf.reduce_max(pot)
        return ((cbed,probe),pot)

    def _parse_tfr_element_multitask(self, element):
        parse_dic = {
            'cbed_feature': tf.io.FixedLenFeature([], tf.string), 
            'probe_feature': tf.io.FixedLenFeature([], tf.string),
            'pot_feature': tf.io.FixedLenFeature([], tf.string),
            }
        example_message = tf.io.parse_single_example(element, parse_dic)

        cbed_feature = example_message['cbed_feature'] 
        probe_feature = example_message['probe_feature']
        pot_feature = example_message['pot_feature']
        cbed = tf.io.parse_tensor(cbed_feature, out_type=tf.float32)
        cbed.set_shape([250, 250, 1])
        cbed = tf.image.resize(cbed, (256,256))
        probe = tf.io.parse_tensor(probe_feature, out_type=tf.float32)
        probe.set_shape([250, 250, 1])
        probe = tf.image.resize(probe, (256,256))
        pot = tf.io.parse_tensor(pot_feature, out_type=tf.float32)
        pot.set_shape([250, 250, 1])
        pot = tf.image.resize(pot, (256,256))
        pot_max = tf.reduce_max(pot)
        pot = pot/tf.reduce_max(pot)
        pot_max.set_shape([None])
        return ((cbed,probe),(pot, pot_max))

    def data_augmentation(self, inputs):
        #np.random.seed(0)
        #sigma_blur = 0.5*np.random.uniform(low=0.1, high=5)
        counts_per_pixel = np.random.randint(low = 100, high = 10000)
        augment =augmentation.image_augmentation(backgrnd = False) 
        augment.set_params(shot=True, counts_per_pixel=counts_per_pixel)
        images = inputs[0]
        outputs = (augment.augment_img(images), inputs[1])
    
        return outputs


'''
=====================================================================================================================================
'''
   
'''
These functions are to create the convolution layers
'''

# %%
#Adapted from get_cross_correlation algorithm in py4DStem (corrPower = 0)

def cross_correlate(x):
    cb = x[0]
    pr = x[1]
    cb = tf.keras.backend.permute_dimensions(cb, (3,0,1,2))
    pr = tf.keras.backend.permute_dimensions(pr, (3,0,1,2))
    
    #shift probe to the origin
    pr = tf.signal.ifftshift(pr, axes = (2,3))
    cbed = tf.signal.fft2d(tf.cast(cb,tf.complex64))
    probe = tf.signal.fft2d(tf.cast(pr,tf.complex64))
    
    ccff = tf.multiply(cbed, tf.math.conj(probe))
    cc = tf.math.real(tf.signal.ifft2d(ccff))
    cc = tf.keras.backend.permute_dimensions(cc, (1,2,3,0))
    
    #normalize each cross-corr
    cc, _ = tf.linalg.normalize(cc, axis = (1,2))                 
        
    return cc


def cross_correlate_ff(x):
    cb = x[0]
    pr = x[1]
    cb = tf.keras.backend.permute_dimensions(cb, (3,0,1,2))
    pr = tf.keras.backend.permute_dimensions(pr, (3,0,1,2))
    
    #shift probe to the origin
    pr = tf.signal.ifftshift(pr, axes = (2,3))
    cbed = tf.signal.fft2d(tf.cast(cb,tf.complex64))
    probe = tf.signal.fft2d(tf.cast(pr,tf.complex64))
    
    ccff = tf.multiply(cbed, tf.math.conj(probe))
    #cc = tf.math.real(tf.signal.ifft2d(ccff))
    ccff = tf.keras.backend.permute_dimensions(ccff, (1,2,3,0))
    
    #normalize each cross-corr
    ccff, _ = tf.linalg.normalize(ccff, axis = (1,2)) 
    ccff_real = tf.math.real(ccff)
    ccff_im = tf.math.imag(ccff)
    ccff = tf.concat([ccff_real,ccff_im], axis = -1)
    
    return ccff


def cross_correlate_iff(x):
    input_channel = x.shape[-1] // 2

    input_complex = tf.dtypes.complex(x[:,:,:,:input_channel], x[:,:,:,input_channel:])
    input_transposed = tf.transpose(input_complex, [0,3,1,2])
    output_complex = tf.math.real(tf.signal.ifft2d(input_transposed))
    output = tf.transpose(output_complex, [0,2,3,1])
    
    return output

# %%
#MCDropout
class MonteCarloDropout(tf.keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)


def convSpec2d_block(input_tensor, n_filters, n_depth=2, input_channel=1, kernel_size = 3, activation = 'relu', dp_rate = 0.1, batchnorm = True):
    """Function to add 2 spectral complex convolutional layers with the parameters passed to it"""
    # Adapt Oren Rippel's paper (https://github.com/oracleofnj/spectral-repr-cnns/tree/master/src/modules)
    
    kl = 'he_uniform'
    if batchnorm:
        x = tf.keras.layers.BatchNormalization()(input_tensor)
    else:
        x = input_tensor
            
    for _ in range(n_depth):
        x = convSpectral.ConvComplex2D(rank=2, filters = n_filters, kernel_size = (kernel_size, kernel_size),
                                padding = 'same', kernel_initializer = kl, bias_initializer = kl, kernel_regularizer= 'l2', bias_regularizer= 'l2')(x)
        if batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)
        x = MonteCarloDropout(dp_rate)(x)
    
    return x


def conv2d_block(input_tensor, n_filters, n_depth=2, input_channel=1, kernel_size = 3, activation = 'relu', dp_rate = 0.1, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    
    kl = 'he_uniform'
    if batchnorm:
        x = tf.keras.layers.BatchNormalization()(input_tensor)
    else:
        x = input_tensor
    
    for _ in range(n_depth):
        x = tf.keras.layers.Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),             
                               padding = 'same', kernel_initializer = kl, bias_initializer = kl, kernel_regularizer= 'l2', bias_regularizer= 'l2')(x)
        if batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)
        x = MonteCarloDropout(dp_rate)(x)
    
    return x
