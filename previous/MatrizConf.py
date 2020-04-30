import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = "Matriz de confusión Normalizada"
        else:
            title = 'Matriz de Confusión sin normalización'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[list(unique_labels(y_true, y_pred))]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de confusión Normalizada")
    else:
        print('Matriz de Confusión sin normalización')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Etiqueta Real',
           xlabel='Etiqueta Predicha')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

print(os.getcwd())
f = open('modelsQuick/Adam_0_001_sigmoid/test', 'r')
f = f.read()
label = np.zeros(np.sum(1 for lin in f.split("\n"))-1)
prediccion = np.zeros(np.sum(1 for lines in f.split("\n"))-1)
etiquetas = ['sheep', 'bear', 'bee', 'cat', 'camel', 'cow', 'crab', 'crocodile', 'duck', 'elephant', 'dog', 'giraffe']
i = 0
# print(f)
bla = True
for line in f.split("\n"):
    if bla:
        bla = False
        continue
    frase = line.split(",")
    if frase == '':
        break
    print(frase)
    label[i] = frase[1]
    print(label[i])
    prediccion[i] = frase[2]
    print(prediccion[i])
    i += 1
print(confusion_matrix(label, prediccion))
plot_confusion_matrix(label, prediccion, classes=etiquetas, normalize=False)
plt.savefig('confusionMatrizQuicknoNorm.pdf', bbox_inches='tight')
plt.show()