#
# Easily accomplish the task of machine learning using scikit-learn (sklearn).
# This is very entry-level code.
#

from sklearn import tree

# Have some data with 2-D tuple and a collection of these tuples.
# Totally it is a 3-D tuple at all.
featrues = [[140, 1, 5],
            [156, 2, 7],
            [121, 9, 3],
            [189, 10, 4]]

# The labels here describe what these features mean.
# 0 -> just the type "0", it could be anything in the real world.
# 1 -> just the type "1", it could be anything in the real world.
labels = [0, 1, 1, 0]

# In this case, you can think of 0 as an orange and 1 as an apple.
# The data in the tuple are: long, wide, and color, respectively.

decision_tree_classifier = tree.DecisionTreeClassifier()
decision_tree_classifier.fit(featrues, labels)


def predict(predicted):
    print(
        "Predict {} -> type {}".format(predicted,
                                       decision_tree_classifier.predict(predicted))
    )


predict([[156, 2, 7]]) # apple 1

predict([[146, 1, 7]]) # orange 0

predict([[190, 12, 3]]) # orange 0

predict([[123, 5, 2]]) # apple 1
