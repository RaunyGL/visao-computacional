import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(img_height=180, img_width=180):
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(101, activation='softmax')  # 101 classes no Food-101
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model
