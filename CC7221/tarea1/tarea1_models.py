"""
@author: martinvillesca

Module containing models
"""
import tensorflow as tf
from tensorflow.keras import layers

class VGG_Block(layers.Layer):
    def __init__(self,
                 units=64,
                 kernel=3,
                 name='VGG_Block',
                 **kwargs):
        
        super(VGG_Block, self).__init__(name=name, **kwargs)

        self.relu = layers.ReLU()

        self.conv1 = layers.Conv2D(units, kernel, padding='valid', kernel_initializer='he_normal')
        self.bn_conv_1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(units, kernel, padding='valid', kernel_initializer='he_normal')
        self.bn_conv_2 = layers.BatchNormalization()
        
        self.conv3 = layers.Conv2D(units, kernel, padding='valid', kernel_initializer='he_normal')
        self.bn_conv_3 = layers.BatchNormalization()
        
        self.conv4 = layers.Conv2D(units, kernel, padding='valid', kernel_initializer='he_normal')
        self.bn_conv_4 = layers.BatchNormalization()
    
    def call(self, inputs):
        x = self.bn_conv_1(self.conv1(inputs))
        x = self.relu(x)
        x = self.bn_conv_2(self.conv2(inputs))
        x = self.relu(x)
        x = self.bn_conv_3(self.conv3(inputs))
        x = self.relu(x)
        x = self.bn_conv_4(self.conv4(inputs))
        x = self.relu(x)
        return x


class BasicModel(tf.keras.Model):
    def __init__(self, name='BasicVGG', **kwargs):
        super(BasicModel, self).__init__(name=name, **kwargs)
        
        self.conv1 = layers.Conv2D(64, 3, strides=2, activation='relu')
        self.max_pool = layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='valid')

        self.block1 = VGG_Block(units=64,name='block1')
        self.block2 = VGG_Block(units=64,name='block2')
        self.block3 = VGG_Block(units=128,name='block3')
        self.block4 = VGG_Block(units=128,name='block4')
        self.block5 = VGG_Block(units=256,name='block5')

        self.fc = layers.Dense(units=250, activation='relu', kernel_initializer='he_normal')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.max_pool(self.block1(x))
        x = self.max_pool(self.block2(x))
        x = self.max_pool(self.block3(x))
        x = self.max_pool(self.block4(x))
        x = self.block5(x)
        
        x = layers.Flatten()(x)
        x = self.fc(x)

        return x