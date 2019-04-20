import enum
import numpy
import scipy
import time


from AIToolbox import MDP, POMDP
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, auc


class Action(enum.Enum):
    ANALIZE = 0
    BLOCK   = 1
    PASS    = 2


class Observe(enum.Enum):
    OK  = 0
    BAD = 1


class State(enum.Enum):
    REGULAR = 0
    TUNNEL  = 1


def create_model(Q, L):
    S = 2
    A = 3
    O = 2

    # In case of DNS tunneling model:
    #
    # States:
    #  1. Block request
    #  2. Pass request
    #
    # Actions:
    #  1. Retrieve the data from detector.
    #  2. Block the request (domain?).
    #  3. Unblock the request (domain?).
    #
    # Observations:
    #  1. Detector labels (what, domain or request?) as safe.
    #  2. Detector labels (what, domain or request?) as unsafe.
    #
    # Rewards:
    #  1. Accessing detector.
    #  2. Labeling domain (request?) as safe.
    #  3. Labeling domain (request?) as unsafe.
    model = POMDP.Model(O, S, A)

    transitions = [[[0 for x in xrange(S)] for y in xrange(A)] for k in xrange(S)]
    rewards = [[[0 for x in xrange(S)] for y in xrange(A)] for k in xrange(S)]
    observations = [[[0 for x in xrange(O)] for y in xrange(A)] for k in xrange(S)]

    # Transitions
    # If we listen, nothing changes.
    for s in xrange(S):
        transitions[s][Action.ANALIZE.value][s] = 1.0

    # If we pick a door, tiger and treasure shuffle.
    for s in xrange(S):
        for s1 in xrange(S):
            transitions[s][Action.BLOCK.value][s1] = 1.0 / S
            transitions[s][Action.PASS.value ][s1] = 1.0 / S

    # Observations
    # If we listen, we guess right 85% of the time.
    observations[Observe.OK.value ][Action.ANALIZE.value][Observe.OK.value ] = Q
    observations[Observe.OK.value ][Action.ANALIZE.value][Observe.BAD.value] = 1-Q

    observations[Observe.BAD.value][Action.ANALIZE.value][Observe.BAD.value] = Q
    observations[Observe.BAD.value][Action.ANALIZE.value][Observe.OK.value ] = 1-Q

    # Otherwise we get no information on the environment.
    for s in xrange(S):
        for o in xrange(O):
            observations[s][Action.BLOCK.value][o] = 1.0 / O
            observations[s][Action.PASS.value ][o] = 1.0 / O

    # Rewards
    # Listening has a small penalty
    for s in xrange(S):
        for s1 in xrange(S):
            rewards[s][Action.ANALIZE.value][s1] = L*0.8

    # Treasure has a decent reward, and tiger a bad penalty.
    for s1 in xrange(S):
        # Put higher rewards for accurate decisions.
        # True negative.
        rewards[Observe.BAD.value][Action.BLOCK.value][s1] = L
        # True positive.
        rewards[Observe.OK.value ][Action.PASS.value][s1] = L

        # False negative.
        rewards[Observe.OK.value ][Action.BLOCK.value][s1] = -(1-L)
        # False positive.
        rewards[Observe.BAD.value][Action.PASS.value][s1] = (1-L)

    model.setTransitionFunction(transitions)
    model.setRewardFunction(rewards)
    model.setObservationFunction(observations)

    return model


def estimate(Q, L):
    # Create model of the problem.
    model = create_model(Q, L)
    model.setDiscount(0.95)

    # Set the horizon. This will determine the optimality of the policy
    # dependent on how many steps of observation/action we plan to do. 1 means
    # we're just going to do one thing only, and we're done. 2 means we get to
    # do a single action, observe the result, and act again. And so on.
    horizon = 1000
    # The 0.0 is the tolerance factor, used with high horizons. It gives a way
    # to stop the computation if the policy has converged to something static.
    solver = POMDP.IncrementalPruning(horizon, 0.0)

    # Solve the model. After this line, the problem has been completely
    # solved. All that remains is setting up an experiment and see what
    # happens!
    solution = solver(model)

    # We create a policy from the solution, in order to obtain actual actions
    # depending on what happens in the environment.
    policy = POMDP.Policy(2, 3, 2, solution[1])

    # We begin a simulation, we start from a uniform belief, which means that
    # we have no idea on which side the tiger is in. We sample from the belief
    # in order to get a "real" state for the world, since this code has to
    # both emulate the environment and control the agent. The agent won't know
    # the sampled state though, it will only have the belief to work with.
    b = [0.5, 0.5]
    s = 0

    # The first thing that happens is that we take an action, so we sample it now.
    a, ID = policy.sampleAction(b, horizon)

    # We loop for each step we have yet to do.
    totalReward = 0.0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    total = 0

    y_true = []
    y_pred = []

    for t in xrange(horizon - 1, -1, -1):
        #print("state = %s, action = %s" % (State(s), Action(a)))
        # We advance the world one step (the agent only sees the observation
        # and reward).
        s1, o, r = model.sampleSOR(s, a)
        # We update our total reward.
        totalReward += r

        # Blcok a tunnel.
        if Action(a) == Action.BLOCK and State(s) == State.TUNNEL:
            TP += 1

        # Pass regular.
        if Action(a) == Action.PASS and State(s) == State.REGULAR:
            TN += 1

        # Block actually normal tunnel.
        if Action(a) == Action.BLOCK and State(s) == State.REGULAR:
            FN += 1

        # Pass tunnel.
        if Action(a) == Action.PASS and State(s) == State.TUNNEL:
            FP += 1

        if Action(a) in (Action.BLOCK, Action.PASS):
            total += 1
            y_true.append(s)
            y_pred.append(0 if Action(a) == Action.PASS else 1)

        # We explicitly update the belief to show the user what the agent is
        # thinking. This is also necessary in some cases (depending on
        # convergence of the solution, see below), otherwise its only for
        # rendering purpouses. It is a pretty expensive operation so if
        # performance is required it should be avoided.
        b = POMDP.updateBelief(model, b, a, o)

        # Now that we have rendered, we can use the observation to find out
        # what action we should do next.
        #
        # Depending on whether the solution converged or not, we have to use
        # the policy differently. Suppose that we planned for an horizon of 5,
        # but the solution converged after 3. Then the policy will only be
        # usable with horizons of 3 or less. For higher horizons, the highest
        # step of the policy suffices (since it converged), but it will need a
        # manual belief update to know what to do.
        #
        # Otherwise, the policy implicitly tracks the belief via the id it
        # returned from the last sampling, without the need for a belief
        # update. This is a consequence of the fact that POMDP policies are
        # computed from a piecewise linear and convex value function, so
        # ranges of similar beliefs actually result in needing to do the same
        # thing (since they are similar enough for the timesteps considered).
        if t > policy.getH():
            a, ID = policy.sampleAction(b, policy.getH())
        else:
            a, ID = policy.sampleAction(ID, o, t)

        s = s1

    accuracy = float(TP+TN)/total
    print("Accuracy(%f): %f" % (L, accuracy))

    return dict(
        accuracy=accuracy,
        fn=float(FN)/total,
        tn=float(TN)/total,
        fp=float(FP)/total,
        tp=float(TP)/total,
        y_true=y_true,
        y_pred=y_pred,
    )


def plot(Q, cm=plt.cm.magma, dpi=600):
    inputs = numpy.arange(0.2, 1.02, 0.1)
    num_inputs = len(inputs)

    metrics = numpy.array([estimate(Q, l) for l in inputs])

    lines = [Line2D([0], [0], color=cm(l))
             for l in numpy.linspace(0, 0.9, num_inputs-1)]

    orange_color = (0.9867, 0.535582, 0.38221, 1.0)

    # A sub-plot for ROC curves.
    _, ax = plt.subplots()
    for i in range(num_inputs-1):
        # Calculate ROC for each experiment.
        y_true, y_pred= metrics[i]["y_true"], metrics[i]["y_pred"]
        fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)

        # Set the equal dimensions to the returned metrics.
        roc_auc = auc(fpr, tpr)

        label = "ROC for L={0} (AUC = {1:.2})".format(inputs[i], roc_auc)
        ax.plot(fpr, tpr, alpha=0.7, c=lines[i].get_color(), label=label)

    ax.plot([0,1], [0,1], linestyle="--", lw=2, color=orange_color,
            alpha=0.8, label="Luck")

    ax.legend(loc="lower right")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.figure.savefig("images/roc.png", dpi=dpi)

    # Render accuracy depending on security level.
    _, ax = plt.subplots()
    l1, = ax.plot(inputs, [r["accuracy"] for r in metrics], marker="o")

    l2, = ax.plot(inputs, [Q]*num_inputs, c="grey", linestyle="dashed")
    ax.legend([l1, l2], ["POMDP detector", "ML detector"])
    ax.set_xlabel("L")
    ax.set_ylabel("Accuracy")
    ax.figure.savefig("images/acc.png", dpi=dpi)

    _, ax = plt.subplots()
    ax.stackplot(inputs,
                 [[r["fn"] for r in metrics],
                  [r["tn"] for r in metrics],
                  [r["fp"] for r in metrics],
                  [r["tp"] for r in metrics]],
                 labels=["False negative", "True negative",
                         "False positive", "True positive"],
                 colors=[lines[i+3].get_color() for i in range(4)])

    ax.legend(loc="lower right")
    ax.set_xlabel("L")
    ax.set_ylabel("Rate")
    ax.figure.savefig("images/met.png", dpi=dpi)


def main():
    plot(Q=0.85)


if __name__ == "__main__":
    main()
