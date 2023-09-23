import numpy as np
from sklearn import metrics
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

def Intersection_over_Union(confusion_matrix):

    intersection = np.diag(confusion_matrix)
    union = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)
    IoU = intersection / union
    return IoU

def Mean_Intersection_over_Union(confusion_matrix):
    intersection = np.diag(confusion_matrix)
    union = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)
    IoU = intersection / union
    MIoU = np.mean(IoU)
    return MIoU

names = ['normal', 'benign', 'insitu', 'invasive']

def plot_confusion_matrix(cm,path):

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cm / np.sum(cm,axis=1), annot=True, fmt='.1%', cmap='Blues')
    ax.set_title('Confusion Matrix\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');
    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(names)
    ax.yaxis.set_ticklabels(names)
    ## Display the visualization of the Confusion Matrix.
    plt.savefig(path + 'confusion_matrix.png')
    plt.show()

def evaluation(preds, gts, path):

    print('start evaluation............')
    for i, (pred, gt) in enumerate(zip(preds, gts)):
        pred = np.ndarray.flatten(pred)
        gt = np.ndarray.flatten(gt)
        if i == 0:
            Pred = pred
            GT = gt
        else:
            Pred = np.concatenate((Pred,pred),axis=0)
            GT = np.concatenate((GT,gt),axis=0)

    #calculate custom score
    a = 0
    b = 0
    for i in range(Pred.shape[0]):
        a = a + abs(Pred[i] - GT[i])
        if Pred[i] + GT[i] != 0:
            b = b + max(GT[i], abs(GT[i] - 3))
    print('custom score:', 1 - a * 1.0 / b)
    
    cm = metrics.confusion_matrix(GT, Pred)
    IoU = Intersection_over_Union(cm)
    MIoU = np.mean(IoU)
    print('Normal IoU:',IoU[0])
    print('Benign IoU:', IoU[1])
    print('Insitu IoU:', IoU[2])
    print('Invasive IoU:', IoU[3])
    print('MioU:', MIoU)
    print(classification_report(GT, Pred, target_names=names))
    # plot_confusion_matrix(cm, path)

    file = open(path + 'metrics.txt', "w")
    file.write("custom score is {} \n".format(1 - a * 1.0 / b))
    file.write("Normal IoU is {} \n".format(IoU[0]))
    file.write("Benign IoU is {} \n".format(IoU[1]))
    file.write("Insitu IoU is {} \n".format(IoU[2]))
    file.write("Invasive IoU is {} \n".format(IoU[3]))
    file.write("MIoU is {} \n".format(MIoU))
    file.write("Classification report {} \n".format(classification_report(GT, Pred, target_names=names)))
    file.close()
