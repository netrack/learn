import argparse
import csv
import contextlib
import random
import numpy


def first(tuples):
    return numpy.array([t[0] for t in tuples])


def last(tuples):
    return numpy.array([t[1:] for t in tuples])


def main():
    parser = argparse.ArgumentParser(description="Neural network model.")
    parser.add_argument("total", metavar="TOTAL", type=int,
                        help="Count of items to train.")
    parser.add_argument("batch", metavar="BATCH", type=int,
                        help="Size of a batch.")
    parser.add_argument("negative", metavar="FILE",
                        help="File without DNS-tunneling traffic.")
    parser.add_argument("positive", metavar="FILE",
                        help="File with DNS-tunneling traffic.")

    args = parser.parse_args()

    import keras

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(units=14, activation="relu", input_dim=18))
    # model.add(keras.layers.Dropout(0.7))
    model.add(keras.layers.Dense(units=7, activation="relu"))
    model.add(keras.layers.Dense(units=2, activation="softmax"))

    model.compile(loss="categorical_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])

    with contextlib.ExitStack() as stack:
        neg_file = open(args.negative)
        pos_file = open(args.positive)

        stack.enter_context(neg_file)
        stack.enter_context(pos_file)

        positive_file = csv.reader(neg_file)
        negative_file = csv.reader(pos_file)

        # Read headers of the CSV files.
        print(len(next(positive_file)))
        print(len(next(negative_file)))

        for _ in range(args.total):
            batch = []

            for _ in range(args.batch):
                batch.append(numpy.array(next(positive_file)))
                batch.append(numpy.array(next(negative_file)))

            # Shuffle the training sets and then train a model.
            random.shuffle(batch)
            x_batch = last(batch)
            y_batch = keras.utils.to_categorical(first(batch))

            model.train_on_batch(x_batch, y_batch)

        # Evaluate model accuracy.
        batch = []
        for _ in range(args.batch):
            batch.append(numpy.array(next(positive_file)))
            batch.append(numpy.array(next(negative_file)))

        # Shuffle the training sets and then train a model.
        random.shuffle(batch)
        x_batch = last(batch)
        y_batch = keras.utils.to_categorical(first(batch))

        loss_and_metrics = model.evaluate(x_batch, y_batch)
        print(loss_and_metrics)


if __name__ == "__main__":
    main()
