import argparse
import csv
import contextlib
import random
import numpy
import keras

from keras import backend as K


def first(tuples):
    """Return a numpy array with the first elements of each tuple."""
    return numpy.array([t[0] for t in tuples])


def last(tuples):
    """Return a numpy array with the last elements of each tuple."""
    return numpy.array([t[1:] for t in tuples])


class Recall(keras.layers.Layer):

    def __init__(self, name="recall", **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = K.variable(0, dtype="int32")
        self.false_negatives = K.variable(0, dtype="int32")

    def reset_states(self):
        K.set_value(self.true_positives, 0)
        K.set_value(self.false_negatives, 0)

    def __call__(self, y_true, y_pred):
        y_true = K.cast(y_true, "int32")
        y_pred = K.cast(K.round(y_pred), "int32")
        neg_y_pred = 1 - y_pred

        true_positives = K.sum(y_true * y_pred)
        false_negatives = K.sum(y_true * neg_y_pred)

        current_true_positives = self.true_positives * 1
        current_false_negatives = self.false_negatives * 1

        self.add_update(K.update_add(self.true_positives, true_positives), inputs=[y_true, y_pred])
        self.add_update(K.update_add(self.false_negatives, false_negatives), inputs=[y_true, y_pred])

        tp = current_true_positives + true_positives
        fn = current_false_negatives + false_negatives

        return tp / (tp + fn)


class Precision(keras.layers.Layer):

    def __init__(self, name="precision", **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = K.variable(0, dtype="int32")
        self.false_positives = K.variable(0, dtype="int32")

    def reset_states(self):
        K.set_value(self.true_positives, 0)
        K.set_value(self.false_positives, 0)

    def __call__(self, y_true, y_pred):
        y_true = K.cast(y_true, "int32")
        y_pred = K.cast(K.round(y_pred), "int32")
        neg_y_true = 1 - y_true

        true_positives = K.sum(y_true * y_pred)
        false_positives = K.sum(neg_y_true * y_pred)

        current_true_positives = self.true_positives * 1
        current_false_positives = self.false_positives * 1

        self.add_update(K.update_add(self.true_positives, true_positives), inputs=[y_true, y_pred])
        self.add_update(K.update_add(self.false_positives, false_positives), inputs=[y_true, y_pred])

        tp = current_true_positives + true_positives
        fp = current_false_positives + false_positives

        return tp / (tp + fp)


def train_and_estimate(dim, num_classes, units, total, batch_size, filenames):
    layer1 = int(units[0])
    layer2 = int(units[2])
    layer3 = int(units[3])
    layer4 = int(units[4])

    recall = Recall()
    precision = Precision()

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(layer1, activation="relu", input_dim=dim))
    model.add(keras.layers.Dropout(units[1]))
    model.add(keras.layers.Dense(layer2, activation="relu"))
    model.add(keras.layers.Dense(layer3, activation="tanh"))
    # model.add(keras.layers.Dense(layer4, activation="relu"))
    model.add(keras.layers.Dropout(units[5]))
    model.add(keras.layers.Dense(num_classes, activation="softmax"))

    model.compile(loss="categorical_crossentropy",
                  optimizer="nadam",
                  metrics=["accuracy", recall, precision])

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
            y_batch = keras.utils.to_categorical(first(batch), num_classes)

            model.train_on_batch(x_batch, y_batch)

        # Evaluate model accuracy.
        batch = []
        for _ in range(batch_size):
            for f in files:
                batch.append(numpy.array(next(f)))

        # Shuffle the training sets and then train a model.
        random.shuffle(batch)
        x_batch = last(batch)
        y_batch = keras.utils.to_categorical(first(batch), num_classes)

        loss_and_metrics = model.evaluate(x_batch, y_batch)

    return loss_and_metrics[1:]


def main():
    parser = argparse.ArgumentParser(description="Neural network model.")
    parser.add_argument("total", metavar="TOTAL", type=int,
                        help="Count of items to train.")
    parser.add_argument("batch", metavar="BATCH", type=int,
                        help="Size of a batch.")
    parser.add_argument("files", metavar="FILE", nargs="+",
                        help="File with DNS traffic attributes.")

    args = parser.parse_args()
    x0 = [128, 0.1, 128, 128, 128, 0.1]

    return train_and_estimate(18, 6, x0, args.total, args.batch, args.files)


if __name__ == "__main__":
    print(main())
