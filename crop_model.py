from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import model_selection
import joblib  # <-- Import joblib

# Load data
crop = pd.read_csv('Data/crop_recommendation.csv')
X = crop.iloc[:, :-1].values
Y = crop.iloc[:, -1].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)

# Define models
models = [
    ('SVC', SVC(gamma='auto', probability=True)),
    ('svm1', SVC(probability=True, kernel='poly', degree=1)),
    ('svm2', SVC(probability=True, kernel='poly', degree=2)),
    ('svm3', SVC(probability=True, kernel='poly', degree=3)),
    ('svm4', SVC(probability=True, kernel='poly', degree=4)),
    ('svm5', SVC(probability=True, kernel='poly', degree=5)),
    ('rf', RandomForestClassifier(n_estimators=21)),
    ('gnb', GaussianNB()),
    ('knn1', KNeighborsClassifier(n_neighbors=1)),
    ('knn3', KNeighborsClassifier(n_neighbors=3)),
    ('knn5', KNeighborsClassifier(n_neighbors=5)),
    ('knn7', KNeighborsClassifier(n_neighbors=7)),
    ('knn9', KNeighborsClassifier(n_neighbors=9)),
]

# Voting classifier
vot_soft = VotingClassifier(estimators=models, voting='soft')
vot_soft.fit(X_train, y_train)
y_pred = vot_soft.predict(X_test)

# Cross-validation and accuracy
scores = model_selection.cross_val_score(vot_soft, X_test, y_test, cv=5, scoring='accuracy')
print("Accuracy: ", scores.mean())

score = accuracy_score(y_test, y_pred)
print("Voting Score % d" % score)

# Save model using joblib
joblib.dump(vot_soft, 'Crop_Recommendation.pkl')
