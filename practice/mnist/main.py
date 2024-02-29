import logging

from keras import Sequential, Model
from keras import layers
from keras.datasets import mnist


logging.basicConfig(level="DEBUG")


def prepare_model(train_images, train_labels) -> Model:
    model = Sequential(
        [
            layers.Dense(512, activation="relu"),
            layers.Dense(10, activation="softmax")
        ]
    )
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.fit(train_images, train_labels, epochs=5, batch_size=128)

    return model


def main() -> None:
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    reshaped_train_images = train_images.reshape((60000, 28 * 28)).astype("float32") / 255
    reshaped_test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255

    model = prepare_model(reshaped_train_images, train_labels)

    sample_size = 10
    test_digits = reshaped_test_images[:sample_size]
    predictions = model.predict(test_digits)

    for index, label in enumerate(test_labels[:sample_size]):
        print(f"Correct answer: {label}")
        print(f"Predicted answer: {predictions[index].argmax()}", end="\n")


if __name__ == "__main__":
    main()
