import argparse
import csv
import contextlib
import random
import numpy


def first(tuples):
    """Return a numpy array with the first elements of each tuple."""
    return numpy.array([t[0] for t in tuples])


def last(tuples):
    """Retur a numpy array with the last elements of each tuple."""
    return numpy.array([t[1:] for t in tuples])


def train_and_estimate(dim, num_classes, units, total, batch_size, filenames):
    import keras

    layer1 = int(units[0])
    layer2 = int(units[2])
    layer3 = int(units[3])

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(layer1, activation="relu", input_dim=dim))
    model.add(keras.layers.Dropout(units[1]))
    model.add(keras.layers.Dense(layer2, activation="relu"))
    model.add(keras.layers.Dense(layer3, activation="relu"))
    model.add(keras.layers.Dropout(units[4]))
    model.add(keras.layers.Dense(num_classes, activation="softmax"))

    model.compile(loss="categorical_crossentropy",
                  optimizer="nadam",
                  metrics=["accuracy"])

    with contextlib.ExitStack() as stack:
        new = lambda name: stack.enter_context(open(name))
        files = [csv.reader(new(name)) for name in filenames]

        # Read headers of the CSV files.
        for f in files:
            next(f)

        # Read specified number of times a batch to train network.
        for _ in range(total):
            batch = []

            for _ in range(batch_size):
                for f in files:
                    batch.append(numpy.array(next(f)))

            # Shuffle the training sets and then train a model.
            random.shuffle(batch)
            x_batch = last(batch)
            y_batch = keras.utils.to_categorical(first(batch))

            model.train_on_batch(x_batch, y_batch)

        # Evaluate model accuracy.
        batch = []
        for _ in range(batch_size):
            for f in files:
                batch.append(numpy.array(next(f)))

        # Shuffle the training sets and then train a model.
        random.shuffle(batch)
        x_batch = last(batch)
        y_batch = keras.utils.to_categorical(first(batch))

        loss_and_metrics = model.evaluate(x_batch, y_batch)
        accuracy = loss_and_metrics[1]

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Neural network model.")
    # parser.add_argument("iterations", metavar="ITERATIONS", type=int,
                        # help="Number of iterations.")
    parser.add_argument("total", metavar="TOTAL", type=int,
                        help="Count of items to train.")
    parser.add_argument("batch", metavar="BATCH", type=int,
                        help="Size of a batch.")
    parser.add_argument("files", metavar="FILE", nargs="+",
                        help="File with DNS traffic attributes.")

    args = parser.parse_args()
    x0 = [12, 0.1, 12, 12, 0.1]

    return train_and_estimate(18, 3, x0, args.total, args.batch, args.files)


if __name__ == "__main__":
    print(main())
