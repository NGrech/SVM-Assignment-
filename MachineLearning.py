from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score, precision_score

datasets = [
    "diabetes.txt",
    "mushrooms.txt",
    "iris.txt"
]

for dset in datasets:
    print("Current Data set: "+dset)
    print("-------------------------")
    X, y = load_svmlight_file(dset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

    experiments = [
        "linear kernel",
        "polynomial degree kernel= 2",
        "polynomial degree kernel= 5",
        "Gaussian kernel gamma = auto",
        "Gaussian kernel gamma = 1",
        "Gaussian kernel gamma = 10"
    ]

    models = [
        SVC(kernel='linear'),
        SVC(kernel='poly', degree=2),
        SVC(kernel='poly', degree=5),
        SVC(kernel='rbf'),
        SVC(kernel='rbf', gamma=1),
        SVC(kernel='rbf', gamma=10)
    ]

    for exp, mod in zip(experiments, models):
        mod.fit(X_train, y_train)
        predictions = mod.predict(X_test)
        print(exp)
        print("-------------------------")
        print("Accuracy = " + str(accuracy_score(y_test, predictions, )))
        if len(set(y_test)) > 2:
            print("Precision = " + str(precision_score(y_test, predictions, average='macro')))
            print("F1 = " + str(f1_score(y_test, predictions, average='macro')))
        else:
            print("Precision = " + str(precision_score(y_test, predictions, average='binary')))
            print("F1 = " + str(f1_score(y_test, predictions, average='binary')))
        print("-------------------------")
