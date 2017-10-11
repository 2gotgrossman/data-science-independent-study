
# coding: utf-8

# In[64]:

import numpy as np
from statistics import mode


# In[39]:

# lines cotains a list of a list of values. The first value in each sub-list is the y_value
def get_formatted_data(data):
    lines = map(lambda x: x.split(" "), data.split("\n"))
    y_vals = []
    x_vals = []
    lines = filter(lambda line : line != [""], lines)
    for line in lines:
        y_vals.append(int(float(line[0])))
        x_vals.append( tuple( map(float, line[1:-1]) ) )
    print(len(y_vals))
    return [x_vals, y_vals]


with open("zip.train", 'r') as f:
    train_data = f.read()
with open("zip.test", 'r') as f:
    test_data = f.read()
train_x_data, train_y_data = get_formatted_data(train_data)
test_x_data, test_y_data = get_formatted_data(test_data)


# In[46]:

def get_distance_metric(vec1, vec2):
    diffs = map(lambda x, y: (x-y)**2, zip(vec1, vec2))
    return sum(diffs)


# In[65]:
def get_label_k_nn(test_vec, train_x_data, train_y_data, k=1):

    distances = map(lambda x: get_distance_metric(test_vec, x), train_x_data)
    
    # finds the indices of the k values that have minimal distance
    # Uses numpy magic
    k = -k
    min_indices = np.argpartition(np.array(distances), k)[k:]
    min_indices = list(min_indices)

    labels = map(lambda x: train_y_data[x], min_indices)
    # Gets the mode
    mode(labels)


# In[70]:

def test_knn(train_x_data, train_y_data, test_x_data, test_y_data, k=1):
    correct = 0
    for test_vec, test_label, i in zip(test_x_data, test_y_data, range(len(test_y_data))):
        prediction = get_label_k_nn(test_vec, train_x_data, train_y_data, k)
        if test_label == prediction:
            print("Correct")
            correct += 1
        else:
            print("False")
    return float(correct) / len(test_y_data)


# In[71]:

test_knn(train_x_data, train_y_data, test_x_data, test_y_data, 1)


import sys
sys.exit(1)

def get_sorted_distance_vec(test_vec, train_x_data, train_y_data):
    distances = map(lambda x: get_distance_metric(test_vec, x), train_x_data)
    # sorts distances by index  
    x = sorted(range(len(distances)), key=lambda k: distances[k])
    return x


# In[101]:

def get_distances_for_all(train_x_data, train_y_data, test_x_data, test_y_data):
    correct = 0
    distance_vectors = []
    for test_vec, test_label, i in zip(test_x_data, test_y_data, range(len(test_y_data))):
        prediction_vec = get_sorted_distance_vec(test_vec, train_x_data, train_y_data)
        distance_vectors.append(prediction_vec)
        if i % 10 ==0:
            print(i)
    return distance_vectors

distance_vectors = get_distances_for_all(train_x_data, train_y_data, test_x_data, test_y_data)

# In[ ]:

for k in range(1, 101)[:1]:
    k_scores = {}
    for i, (vec, true_label) in enumerate(zip(distance_vectors, test_y_data)):
        labels = map(lambda x: train_y_data[x], vec[:k])
        prediction = mode(labels)
        correct = 0
        if prediction == true_label:
            correct += 1
            
    k_scores[k] = float(correct) / float(len(test_y_data))     
print(k_scores)


# In[ ]:




# In[ ]:



