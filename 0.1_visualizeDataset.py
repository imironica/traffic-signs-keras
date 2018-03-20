from util import loadData, \
    displayImagesAndLabels, \
    displayLabelImages, \
    loadDb

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(style='white', context='notebook', palette='deep')

loadDb()

trainDataDir = "Training"
xTrain, yTrain = loadData(trainDataDir)
testDataDir = "Testing"
xTest, yTest = loadData(testDataDir)

print(len(xTrain))
print(len(xTest))

barValues = pd.Series(yTrain).value_counts()
g = sns.countplot(yTrain)
plt.show()

g = sns.countplot(yTest)
plt.show()

displayImagesAndLabels(xTrain, yTrain)
displayImagesAndLabels(xTest, yTest)
for i in range(0, 62):
    displayLabelImages(xTrain, yTrain, i)





