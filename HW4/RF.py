import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree

data = pd.read_csv('Mushroom.csv').to_numpy()
np.random.shuffle(data)
label = data[:,0].reshape((-1,1))
features_list_ = data[:,1:]

sample_size = 6000

feat_train = features_list_[:sample_size, :]
label_train = label[:sample_size, :]

feat_test = features_list_[sample_size:, :]
label_test = label[sample_size:, :]

class RandomForest():
# initialization
  def __init__(self, total_number=100, feats=4):
    self.size = total_number
    self.trees = [0]*total_number
    self.feats = feats
# training part
  def train(self, X, Y):
    sample_size = X.shape[0]
    for i in range(self.size):
      idx = np.random.choice(range(sample_size), sample_size)
      forest_feature = feat_train[idx, :]
      forest_label = label_train[idx, :]
      # specify classifier
      model = tree.DecisionTreeClassifier(criterion='gini', max_depth=2,
                                        max_features=self.feats)
      model = model.fit(forest_feature, forest_label)
      self.trees[i] = model

#  predict part
  def predict(self, X):
    tree_ = 0
    for tree_branch in self.trees:
      #majority vote
      tree_ += tree_branch.predict(X)
      pred = tree_branch.predict(X)
      pred = pred.reshape((-1,1))
    return tree_.reshape((-1,1))

# acc_calculator return accuracy of 2 vecs
def acc_calculator(x, y):
  return sum(x==y) / x.shape[0]

random_forest_ = RandomForest(100, 4)
random_forest_.train(feat_train, label_train)
h_test = np.sign(random_forest_.predict(feat_test))
acc_calculator(h_test, label_test)

# set sizes of features from 5-20 with step 5
feature_array = [5,10,15,20]
size_ = len(feature_array)
train_set_accur = [0] * size_
test_set_accur = [0] * size_
for i in range(size_):
  random_forest_ = RandomForest(100, feature_array[i])
  random_forest_.train(feat_train, label_train)
  h_train = np.sign(random_forest_.predict(feat_train))
  h_test = np.sign(random_forest_.predict(feat_test))
  train_set_accur[i] = acc_calculator(h_train, label_train)
  test_set_accur[i] = acc_calculator(h_test, label_test)

plt.plot(feature_array, train_set_accur, label='train')
plt.plot(feature_array, test_set_accur, label='test')
plt.xlabel('size of features')
plt.ylabel('Scores')
plt.show()



# different numbers of decision trees
trees_list_ = [10, 20, 40, 80, 100]
size_ = len(trees_list_)
train_set_accur = [0] * size_
test_set_accur = [0] * size_
for i in range(size_):
  random_forest_ = RandomForest(total_number=trees_list_[i], feats=20) # fix feats = 20
  random_forest_.train(feat_train, label_train)
  h_train = np.sign(random_forest_.predict(feat_train))
  h_test = np.sign(random_forest_.predict(feat_test))
  train_set_accur[i] = acc_calculator(h_train, label_train)
  test_set_accur[i] = acc_calculator(h_test, label_test)

plt.plot(trees_list_, train_set_accur, label='Train')
plt.plot(trees_list_, test_set_accur, label='Test')
plt.xlabel('Numbers of trees')
plt.ylabel('Scores')
plt.legend()
plt.show()
