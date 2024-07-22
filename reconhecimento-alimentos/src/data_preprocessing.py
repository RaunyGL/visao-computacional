import tensorflow as tf
import os

def load_data(img_height=180, img_width=180, batch_size=32):
    train_dir = os.path.join('data/food-101', 'train')
    val_dir = os.path.join('data/food-101', 'test')

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    
    return train_ds, val_ds
