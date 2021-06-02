from sklearn.metrics import confusion_matrix as confusion


def model_performance(model, train_X, train_y, test_X, test_y):
    print("Train score")
    print(model.score(train_X, train_y))
    print("Test score")
    print(model.score(test_X, test_y))


def confusion_matrix(train_X, train_y, test_X, test_y):
    tn, fp, fn, tp = confusion(train_y, train_X.astype(float)).ravel()

    print("Train")
    print(f"TN: {tn}, TP: {tp}, FN: {fn}, FP: {fp}")

    tn, fp, fn, tp = confusion(test_y, test_X.astype(float)).ravel()

    print("Test")
    print(f"TN: {tn}, TP: {tp}, FN: {fn}, FP: {fp}")
