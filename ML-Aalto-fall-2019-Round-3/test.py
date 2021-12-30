################################ Student Task. Confidence in Classifications.

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import numpy as np

def calculate_accuracy(y, y_hat):
    """
    Calculate accuracy of your prediction

    :param y: array-like, shape=(m, 1), correct label vector
    :param y_hat: array-like, shape=(m, 1), label-vector prediction

    :return: scalar-like, percentual accuracy of your prediction
    """
    ### STUDENT TASK ###
    # YOUR CODE HERE
    correct_predictions = 0
    for i in range (m):
        if y_hat[i] == y[i]:
            correct_predictions = correct_predictions + 1

    return (correct_predictions/m)*100




wine = datasets.load_wine()  # load wine datasets into variable "wine"
X = wine['data']  # matrix containing the feature vectors of wine samples
cat = wine['target'].reshape(-1, 1)  # vector with wine categories (0,1 or 2)

m = cat.shape[0]  # set m equal to the number of rows in features
y = np.zeros((m, 1));  # initialize label vector with zero entries

for i in range(m):
    if (cat[i] == 0):
        y[i, :] = 1  # Class 0
    else:
        y[i, :] = 0  # Not class 0
    #print(y[i,:])

logReg = LogisticRegression(random_state=0)
logReg = logReg.fit(X, y)
y_pred = logReg.predict(X).reshape(-1, 1)

# Tests
test_acc = calculate_accuracy(y, y_pred)

print('Accuracy of the result is: %f%%' % test_acc)

assert 80 < test_acc < 100, "Your accuracy should be above 80% and less than 100%"
assert test_acc < 99, "Your accuracy was too good. You are probably not using correct methods."

print('Sanity check tests passed!')




wine = datasets.load_wine()         # load wine datasets into variable "wine"
X = wine['data']                    # matrix containing the feature vectors of wine samples
y = wine['target'].reshape(-1, 1)   # vector with wine categories (0,1 or 2)

logReg = LogisticRegression(random_state=0,multi_class="ovr") # set multi_class to one versus rest ('ovr')

logReg = logReg.fit(X, y)

y_pred = logReg.predict(X).reshape(-1, 1)

test_acc = calculate_accuracy(y,y_pred)



# make a prediction
# y_probs = ...
# YOUR CODE HERE
#raise NotImplementedError()
y_probs = logReg.predict_proba(X)

# show the inputs and predicted probabilities
# print('first five samples and their probabilities of belonging to classes 0, 1 and 2:')
for i in range(5):
    print("Probabilities of Sample", i+1,':', 'Class 0:',"{:.2f}".format(100*y_probs[i][0],2),'%', 'Class 1:', "{:.2f}".format(100*y_probs[i][1]), '%', 'Class 2:', "{:.2f}".format(100*y_probs[i][2]),'%' )
#
# n_of_discarded_samples = 0

# YOUR CODE HERE
#raise NotImplementedError()
n_of_discarded_samples = 0

for i in range(len(y_probs)):
    #print(y_probs[i][0])
    if y_probs[i][0] < 0.9 :
        n_of_discarded_samples = n_of_discarded_samples + 1
print('Number of discarded samples:', n_of_discarded_samples)