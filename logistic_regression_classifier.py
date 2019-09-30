from sklearn.cross_validation import train_test_split  # for splitting train and test data
from sklearn.linear_model import LogisticRegression  # for importing classifier

# Split the dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=0)

# Build and train the Logistic Regression model without passing any parameters
clf = LogisticRegression()
clf.fit(x_train, y_train)

# Check the accuracy using the model.score() method
accuracy = clf.score(x_test, y_test)
print(accuracy)

# Predict target by passing some new input data inside model.predict() method
clf.predict()
