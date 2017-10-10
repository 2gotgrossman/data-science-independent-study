Ensembles are a divide-and-conquer approach used to improve performance. 
The main principle behind ensemble methods is that a group of “weak learners” can 
come together to form a “strong learner”. Each classifier, individually, is a “weak learner,” 
while all the classifiers taken together are a “strong learner”.

 
 Ensemble methods are meta-algorithms that combine several machine learning techniques into one predictive model in order to *decrease variance (bagging), bias (boosting), or improve predictions (stacking)*.
 
 ### Stacking
1. Split the training set into two disjoint sets.
2. Train several base learners on the first part.
3. Test the base learners on the second part.
4. Using the predictions from 3) as the inputs, and the correct responses as the outputs, train a higher level learner.


![alt-text](https://i.stack.imgur.com/RFfqb.png)

` Ensembling. Train 10 neural networks and average their predictions. It’s a fairly trivial technique that results in easy, sizeable performance improvements.

One may be mystified as to why averaging helps so much, but there is a simple reason for the effectiveness of averaging. Suppose that two classifiers have an error rate of 70%. Then, when they agree they are right. But when they disagree, one of them is often right, so now the average prediction will place much more weight on the correct answer.

The effect will be especially strong whenever the network is confident when it’s right and unconfident when it’s wrong. 
 - Ilya Sutskever `

![alt-text](http://cdn2.hubspot.net/hubfs/2575516/Imported_Blog_Media/skitch.png?t=1506992243557)

Trees and Forests. The random forest starts with a standard machine learning technique called a “decision tree” which, in ensemble terms, corresponds to our weak learner. In a decision tree, an input is entered at the top and as it traverses down the tree the data gets bucketed into smaller and smaller sets. For details see here, from which the figure below is taken.

1. Sample N cases at random with replacement to create a subset of the data (see top layer of figure above). The subset should be about 66% of the total set.
2. At each node:
  + For some number m (see below), m predictor variables are selected at random from all the predictor variables.
  + The predictor variable that provides the best split, according to some objective function, is used to do a binary split on that node.
3. At the next node, choose another m variables at random from all predictor variables and do the same.

Depending upon the value of m, there are three slightly different systems:
1. Random splitter selection: m =1
2. Breiman’s bagger: m = total number of predictor variables
3. Random forest: m << number of predictor variables. Brieman suggests three possible values for m: ½√m, √m, and 2√m



