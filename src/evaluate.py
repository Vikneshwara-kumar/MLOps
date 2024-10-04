import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


labels = ['Resting', 'Low Strain','Medium Strain','Max Strain']

def evaluate(model, X_test, y_test):

    predictions = model.predict(X_test)
    predictions_class = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(np.argmax(y_test, axis=1), predictions_class)*100
    print(accuracy)
    return accuracy


def metric(y_test, classes):
    cm = confusion_matrix(np.argmax(y_test, axis=1), classes)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    print(classification_report(y_test, classes))