import tensorflow as tf
from data_preprocessing import load_data
from model import build_model

def train_model(epochs=10):
    train_ds, val_ds = load_data()

    model = build_model()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    model.save('models/food_classifier_model.h5')

if __name__ == "__main__":
    train_model()
