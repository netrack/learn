import keras


def main():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(units=10, activation="relu", intput_dim=18))
    model.add(keras.layers.Dense(units=5, activation="softmax"))
    model.add(keras.layers.Dropout(0.2))

    model.compile(loss="categorical_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])


if __name__ == "__name__":
    main()
