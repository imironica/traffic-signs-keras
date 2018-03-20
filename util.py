import os
import numpy as np
import skimage.data
import skimage.transform
import matplotlib.pyplot as plt
import zipfile
import urllib.request
import shutil
from sklearn.metrics import confusion_matrix, accuracy_score
import itertools

def unzipFile(fileToUnzip, folderToUnzip):
    """Unzip a file to a specific folder

    fileToUnzip: file to unzip.
    folderToUnzip: new location for the unzipped files.
    """
    with zipfile.ZipFile(fileToUnzip, "r") as zip_ref:
        zip_ref.extractall(folderToUnzip)

def loadDb():
    """Download the database"""

    folderDb = os.path.dirname(os.path.abspath(__file__))
    linkTraining = 'http://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Training.zip'
    linkTestDb = 'http://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Testing.zip'

    trainingZip = 'BelgiumTSC_Training.zip'
    testZip = 'BelgiumTSC_Testing.zip'

    if not os.path.exists(trainingZip):
        print("Downloading {}".format(trainingZip))
        # Download the file from `url` and save it locally under `file_name`:
        with urllib.request.urlopen(linkTraining) as response, open(trainingZip, 'wb') as outFile:
            shutil.copyfileobj(response, outFile)

        print("Unzip {}".format(trainingZip))

        unzipFile(trainingZip, folderDb)

    if not os.path.exists(testZip):
        print("Downloading {}".format(testZip))
        # Download the file from `url` and save it locally under `file_name`:
        with urllib.request.urlopen(linkTestDb) as response, open(testZip, 'wb') as outFile:
            shutil.copyfileobj(response, outFile)

        print("Unzip {}".format(testZip))
        unzipFile(testZip, folderDb)

def loadData(dataDir, resize=False, size = (32, 32)):
    """Loads a data set and returns two lists:

    images: a list of Numpy arrays, each representing an image.
    labels: a list of numbers that represent the images labels.
    """
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(dataDir)
                   if os.path.isdir(os.path.join(dataDir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(dataDir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        for f in file_names:
            if resize:
                image = skimage.transform.resize(skimage.data.imread(f), size)
                images.append(image)
            else:
                images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels


def readDatabase(size=(32, 32)):

    from keras.utils.np_utils import to_categorical
    print('Reading dataset ...')
    xTrain, yTrain = loadData("Training", resize=True, size=size)
    xTest, yTest = loadData("Testing", resize=True, size=size)

    # Preprocess the training data
    labelsCount = len(set(yTrain))
    yTrainCategorical = to_categorical(yTrain, num_classes=labelsCount)
    yTestCategorical = to_categorical(yTest, num_classes=labelsCount)

    # Scale between 0 and 1
    xTrain = np.array(xTrain)
    xTest = np.array(xTest)
    xTrain = xTrain / 255.0
    xTest = xTest / 255.0

    return xTrain, yTrainCategorical, xTest, yTestCategorical, yTest

def displayImagesAndLabels(images, labels):
    """Display the first image of each label."""
    unique_labels = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1
    for label in unique_labels:
        # Pick the first image for each label.
        image = images[labels.index(label)]
        plt.subplot(4, 16, i)  # A grid of 4 rows x 16 columns
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        _ = plt.imshow(image)
    plt.show()

def displayLabelImages(images, labels, label):
    """Display images of a specific label."""
    limit = 24  # show a max of 24 images
    plt.figure(figsize=(15, 5))
    i = 1

    start = labels.index(label)
    end = start + labels.count(label)
    for image in images[start:end][:limit]:
        plt.subplot(3, 8, i)  # 3 rows, 8 per row
        plt.axis('off')
        i += 1
        plt.imshow(image)
    plt.show()


def plotConfusionMatrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def showConfusionMatrix(yLabels, predictedValues):
    predictedLabels = np.argmax(predictedValues, axis=1)

    accuracy = accuracy_score(y_true=yLabels, y_pred=predictedLabels)
    matrix = confusion_matrix(y_true=yLabels, y_pred=predictedLabels)
    print(matrix.shape)
    plotConfusionMatrix(matrix,
                        classes=[str(i) for i in range(0, 62)],
                        title='Confusion matrix')

