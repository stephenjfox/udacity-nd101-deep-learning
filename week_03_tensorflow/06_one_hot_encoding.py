import numpy as np
from sklearn import preprocessing
# we'll represent the labels as one-hot encoded vectors
# these vectors are as long as there are classes ("choices" of output)

# labels (dummy data)
labels = np.array([1,5,3,2,1,4,2,1,3])

print("Labels:", labels)

# Create the encoder
lb = preprocessing.LabelBinarizer()

# find classes and assign (internal fields) as one-hot vectors
lb.fit(labels)

# show the labels as their one-hot equivalents
print("Encodings:")
print(lb.transform(labels))
