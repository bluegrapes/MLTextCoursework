# Homework with Yelp reviews data

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Task 1
#  Read yelp.csv into a Pandas DataFrame and examine it.
df = pd.read_csv("data/yelp.csv")

# Task 2
# Create a new DataFrame that only contains the 5-star and 1-star reviews.
df1 = df[(df.stars == 5) | (df.stars == 1)]

# Task 3
# Define X and y from the new DataFrame, and then split X and y into training and
# testing sets, using the review text as the only feature and the star rating as
# the response.
X = df1.text
y = df1.stars
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Task 4
# Use CountVectorizer to create document-term matrices from X_train and X_test.
vect = CountVectorizer()
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)
X_test_dtm = vect.transform(X_test)

# Task 5
# Use Multinomial Naive Bayes to predict the star rating for the reviews in
# the testing set, and then calculate the accuracy and print the confusion matrix.
model = MultinomialNB()
model.fit(X_train_dtm, y_train)
y_pred_class = model.predict(X_test_dtm)
print("Accuracy score:")
print(metrics.accuracy_score(y_test, y_pred_class))
print()
print("Confusion matrix:")
print(metrics.confusion_matrix(y_test, y_pred_class))
print()

# Task 6 (Challenge)
# Calculate the null accuracy, which is the classification accuracy that could be
# achieved by always predicting the most frequent class.
print("Null accuracy:")
print(y_test.value_counts().head(1) / len(y_test))
print()

# Task 7 (Challenge)
# Calculate which 5 tokens are the most predictive of 5-star reviews, and
# which 5 tokens are the most predictive of 1-star reviews.
X_train_tokens = vect.get_feature_names()
rdf = pd.DataFrame({'token':X_train_tokens,
                    '1star':model.feature_count_[0],
                    '5star':model.feature_count_[1],
                    })
print("5 predictive tokens for 5-star reviews")
print(rdf.sort('5star', ascending=False).head()[["5star", "token"]])
print()
print("5 predictive tokens for 1-star reviews")
print(rdf.sort('1star', ascending=False).head()[["1star", "token"]])
print()

# Task 8 (Challenge)

# Task 9 (Challenge)
