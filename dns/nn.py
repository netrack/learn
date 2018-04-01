import argparse
import csv
import contextlib
import random
import numpy
import os.path
import keras
import scipy

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
from sklearn.metrics import roc_curve, auc
from sklearn import manifold
from keras import backend as K


DPI = 600


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

        # Label for each class.
        labels = [os.path.dirname(name) for name in filenames]

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
        for _ in range(batch_size*4):
            for f in files:
                batch.append(numpy.array(next(f)))

        # Shuffle the training sets and then train a model.
        random.shuffle(batch)
        x_batch = last(batch)
        y_true = first(batch)

        y_batch = keras.utils.to_categorical(y_true, num_classes)
        loss_and_metrics = model.evaluate(x_batch, y_batch)

        y_score = numpy.array(model.predict_proba(x_batch))
        y_batch = numpy.array(y_batch)

    return loss_and_metrics[1:], x_batch, y_batch, y_score, labels


def plot(x_true, y_true, y_score, labels, num_classes, cm=plt.cm.magma):
    tprs, aucs = [], []
    mean_fpr = numpy.linspace(0, 1, 100)

    classes = [Line2D([0], [0], color=cm(l))
               for l in numpy.linspace(0, 1, num_classes-1)]

    orange_color = (0.9867, 0.535582, 0.38221, 1.0)

    # A sub-plot for ROC curves.
    _, ax = plt.subplots()
    for i in range(num_classes-1):
        # Calculate ROC for each class.
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)

        tprs.append(scipy.interp(mean_fpr, fpr, tpr))
        aucs.append(roc_auc)

        label = "ROC for {0} (AUC = {1:.2})".format(labels[i], roc_auc)
        ax.plot(fpr, tpr, alpha=0.3, c=classes[i].get_color(), label=label)

    # Calculate mean ROC for all classes.
    mean_tpr = numpy.mean(tprs, axis=0)
    mean_tpr[0], mean_tpr[-1] = 0, 1
    mean_auc = auc(mean_fpr, mean_tpr)

    # Standard deviation of mean true positives.
    std_tpr = numpy.std(tprs, axis=0)
    std_auc = numpy.std(aucs)
    tprs_upper = numpy.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = numpy.maximum(mean_tpr - std_tpr, 0)

    ax.plot(mean_fpr, mean_tpr, color="black", lw=2, alpha=0.8,
            label="Mean ROC (AUC = {0:.2} $\pm$ {1:.2})".format(
                mean_auc, std_auc))
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper,
                    color="grey", alpha=0.2, label="$\pm$ 1 std. dev.")
    ax.plot([0,1], [0,1], linestyle="--", lw=2, color=orange_color,
            alpha=0.8, label="Luck")

    ax.legend(loc="lower right")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.figure.savefig("images/roc.png", dpi=DPI)

    # Caclulate manifold embedding for the samples space using t-SNE
    # algorithm.
    learn = manifold.TSNE(perplexity=35)
    color = numpy.argmax(y_true, axis=1)

    x_true = learn.fit_transform(x_true)
    noise  = numpy.random.normal(0, 3.0, x_true.shape)
    x_true = x_true + noise

    _, ax = plt.subplots()
    ax.scatter(x_true[:,0], x_true[:,1], marker="x", cmap=cm, c=color)
    ax.legend(classes, labels)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.figure.savefig("images/tsne.png", dpi=DPI)


def main():
    parser = argparse.ArgumentParser(description="Neural network model.")
    parser.add_argument("total", metavar="TOTAL", type=int,
                        help="Count of items to train.")
    parser.add_argument("batch", metavar="BATCH", type=int,
                        help="Size of a batch.")
    parser.add_argument("files", metavar="FILE", nargs="+",
                        help="File with DNS traffic attributes.")

    args = parser.parse_args()
    num_classes = 6

    x0 = [128, 0.1, 128, 128, 128, 0.1]

    metrics, x_true, y_true, y_score, labels = train_and_estimate(
        18, num_classes, x0, args.total, args.batch, args.files)

    print(metrics)
    with open("metrics.txt", "w") as f:
        f.write("accuracy = {0}\n".format(metrics[0]))
        f.write("recall = {0}\n".format(metrics[1]))
        f.write("precision = {0}\n".format(metrics[2]))

    plot(x_true, y_true, y_score, labels, num_classes)


if __name__ == "__main__":
    main()
