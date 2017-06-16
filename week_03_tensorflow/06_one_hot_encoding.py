import numpy as np
from sklearn import preprocessing
# we'll represent the labels as one-hot encoded vectors
# these vectors are as long as there are classes ("choices" of output)

def to_one_hot(array):
    """
    Convert an array - assumed to be labels of a data set - to one-hot vectors,
    for ease of use in a neural network classifier
    """
    lb = preprocessing.LabelBinarizer()

    # find classes and assign (internal fields) as one-hot vectors
    lb.fit(array)

    # show the labels as their one-hot equivalents
    return lb.transform(array)


# labels (dummy data)
label_backing_data = [1,5,3,2,1,4,2,1,3]
labels = np.array(label_backing_data)


print("Labels:", labels)

print("Encodings:")
print(to_one_hot(labels))

# does this apply with 0-based indexs, or just association?
other_labels = labels - 1

print("\nOther labels:", other_labels)
print("Other Encodings:")

print(to_one_hot(other_labels))


new_labels = np.append(other_labels, 5)

print("\nNew labels:", new_labels)

print("New Encodings:")
print(to_one_hot(new_labels))
