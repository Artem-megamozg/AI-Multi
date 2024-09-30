import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory

size = 224

training_set = image_dataset_from_directory("aidata/train", shuffle=True, batch_size=32, image_size=(size, size))
val_dataset = image_dataset_from_directory("aidata/val", shuffle=True, batch_size=32, image_size=(size, size))

data_augmentation = keras.Sequential(
    [keras.layers.RandomFlip("horizontal_and_vertical"),
   keras.layers.RandomRotation(0.2),
    ])

mirrored_strategy = tf.distribute.MirroredStrategy()

with mirrored_strategy.scope():
    base_model = tf.keras.applications.EfficientNetV2M(
        weights='imagenet',
        input_shape=(size, size, 3),
        include_top=False)
    base_model.trainable = False
    inputs = keras.Input(shape=(size, size, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.efficientnet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    accuracy = keras.metrics.BinaryAccuracy()
    optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(), metrics=accuracy)
model.fit(training_set, epochs=10000, validation_data=val_dataset)

