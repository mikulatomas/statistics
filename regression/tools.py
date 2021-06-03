from sklearn.metrics import confusion_matrix as confusion
from sklearn.utils import resample


def model_performance(model, train_X, train_y, test_X, test_y):
    print("Train score")
    print(model.score(train_X, train_y))
    print("Test score")
    print(model.score(test_X, test_y))
