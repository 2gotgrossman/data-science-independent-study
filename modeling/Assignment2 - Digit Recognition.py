
# coding: utf-8

# In[1]:

import numpy as np
from collections import Counter  # For mode
from sklearn.metrics import confusion_matrix


# #### Load data

# In[2]:

# lines cotains a list of a list of values. The first value in each sub-list is the y_value
def get_formatted_data(data):
    lines = map(lambda x: x.split(" "), data.split("\n"))
    y_vals = []
    x_vals = []
    lines = filter(lambda line : line != [""], lines)
    for line in lines:
        y_vals.append(int(float(line[0])))
        x_vals.append( tuple( map(float, line[1:-1]) ) )
    print len(y_vals)
    return [x_vals, y_vals]


# In[3]:

with open("zip.train", 'r') as f:
    train_data = f.read()
with open("zip.test", 'r') as f:
    test_data = f.read()
train_x_data, train_y_data = get_formatted_data(train_data)
test_x_data, test_y_data = get_formatted_data(test_data)
print "Observations in training set: ", len(train_y_data)
print "Observations in testing set: ", len(test_y_data)


# #### Some Common Functions
# **_Note: since we can have more than 1 mode, we will just take the first mode we find._**

# In[6]:

def get_distance_metric(vec1, vec2):
    diffs = map(lambda (x,y): (x-y)**2, zip(vec1, vec2))
    return sum(diffs)

def get_mode(arr):
    return Counter(arr).most_common(1)[0][0]


# #### K Nearest Neighbors Algorithm

# In[7]:

def get_label_k_nn(test_vec, train_x_data, train_y_data, k=1):
    distances = map(lambda x: get_distance_metric(test_vec, x), train_x_data)
    
    # finds the indices of the k values that have minimal distance
    # Uses numpy magic
    min_indices = np.argpartition(np.array(distances), k)[:k]
    labels = map(lambda x: train_y_data[x], min_indices)
    # Gets the mode
    return get_mode(labels)


# In[8]:

def test_knn(train_x_data, train_y_data, test_x_data, test_y_data, k=1):
    correct = 0
    for test_vec, test_label, i in zip(test_x_data, test_y_data, range(len(test_y_data))):
        prediction = get_label_k_nn(test_vec, train_x_data, train_y_data, k)
        if test_label == prediction:
            correct += 1
    return float(correct) / len(test_y_data)


# #### Test for K=1
# It takes about .5 seconds per test example (15 minutes total). The expensive part is generating the `distances` vector.

# In[163]:

test_knn(train_x_data, train_y_data, test_x_data, test_y_data, 1)


# #### Since it takes so long to do this K-NN, we're going to try something slightly different. 
# We're going to create a sorted list of the indices of the distances for each test observation. Then we can find the optimal K pretty quickly

# In[9]:

def get_sorted_distance_vec(test_vec, train_x_data, train_y_data):
    distances = map(lambda x: get_distance_metric(test_vec, x), train_x_data)
    # sorts distances by index  
    x = sorted(range(len(distances)), key=lambda k: distances[k])
    return x


# In[10]:

def get_distances_for_all(train_x_data, train_y_data, test_x_data, test_y_data):
    correct = 0
    distance_vectors = []
    for test_vec, test_label, i in zip(test_x_data, test_y_data, range(len(test_y_data))):
        prediction_vec = get_sorted_distance_vec(test_vec, train_x_data, train_y_data)
        distance_vectors.append(prediction_vec)
    return distance_vectors


# In[ ]:

distance_vectors = get_distances_for_all(train_x_data, train_y_data, test_x_data, test_y_data)


# #### For the first 100 values of K, get K-NN

# In[ ]:

k_scores = []
for k in range(1, 101):
    correct = 0
    for i, (vec, true_label) in enumerate(zip(distance_vectors, test_y_data)):
        # Takes only the first k values and finds their corresponding labels
        labels = map(lambda x: train_y_data[x], vec[:k])
        prediction = get_mode(labels)
        if prediction == true_label:
            correct += 1
    k_scores.append(float(correct) / float(len(test_y_data)))
print k_scores


# #### And the best K is....

# In[ ]:

print "Max K =", np.argmax(k_scores) + 1, "with score of", max(k_scores)


# #### Now, let's find the Confusion Matrix

# In[ ]:

correct = 0
K = 3
predictions = []
for i, (vec, true_label) in enumerate(zip(distance_vectors, test_y_data)):
    labels = map(lambda x: train_y_data[x], vec[:k])
    prediction = get_mode(labels)
    predictions.append(prediction)
confusion_matrix(test_y_data, predictions, labels=range(10))


# In[ ]:



