from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

def baseline_model_score(X_train, y_train, X_test, y_test):
    dummy_clf = DummyClassifier(strategy='most_frequent') # naive classifier, majority vote
    dummy_clf.fit(X_train, y_train)
    y_pred_dummy = dummy_clf.predict(X_test)
    accuracy_dummy = accuracy_score(y_test, y_pred_dummy)

    return accuracy_dummy
