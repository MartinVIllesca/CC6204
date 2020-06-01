import os
import tensorflow as tf
import data
import numpy as np
import tarea1_models as t1

def normalize(image, label):
    """Convert `image` from [0, 255] -> [-0.5, 0.5] floats."""
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    return image, label


if __name__ == "__main__":
    
    tfrecord_location = '/home/martin/Documents/Ramos-Redes/CC7221/tarea1/Sketch_EITZ/'
    train = 'train.tfrecords'
    validation = 'test.tfrecords'
    train_filename = os.path.join(tfrecord_location, train)
    val_filename = os.path.join(tfrecord_location, validation)


    shape_file = os.path.join(tfrecord_location, 'shape.dat')
    mean_file = os.path.join(tfrecord_location, 'mean.dat')

    input_shape = np.fromfile(shape_file, dtype=np.int32)
    mean_image = np.fromfile(mean_file, dtype=np.float32)
    mean_image = np.reshape(mean_image, input_shape)
    number_of_classes = 250
    # print(mean_image.shape)

    tr_dataset = tf.data.TFRecordDataset(train_filename)
    tr_dataset = tr_dataset.map(lambda x : data.parser_tfrecord(x,
                    input_shape, mean_image, number_of_classes))

    # tr_dataset = tr_dataset.map(normalize)
    batch_size = 200
    tr_dataset = tr_dataset.shuffle(5000)
    # tr_dataset = tr_dataset.repeat(2)
    tr_dataset = tr_dataset.batch(batch_size)

    val_dataset = tf.data.TFRecordDataset(val_filename)
    val_dataset = val_dataset.map(lambda x: data.parser_tfrecord(x,
                    input_shape, mean_image, number_of_classes))
    # val_dataset = val_dataset.map(normalize)
    val_dataset = val_dataset.batch(batch_size=200)


    model = t1.BasicModel()

    model.build((1, input_shape[0], input_shape[1], input_shape[2]))
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(tr_dataset,
                        epochs=4,
                        validation_data=val_dataset,
                        validation_steps=20)
