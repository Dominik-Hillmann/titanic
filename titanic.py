import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import scipy.stats as stat

trainData = pd.read_csv(
	'./data/train.csv', 
	na_values = ['']
)
# testData = pd.read_csv(
# 	'./data/test.csv',
# 	na_values = ['']
# )

trainData.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis = 1, inplace = True)
# I drop these attributes since I think that either:
#	- they don't contain information useful to know whether someone survived (Name, Passenger ID)
#	- I don't have the means to dissern useful information from it (all three, Cabin and Ticket because maybe they contain information about where cabin was within the ship

trainData.replace(('male', 'female'), (1, 0), inplace = True)
trainData.columns = [attr if attr != 'Sex' else 'Male' for attr in trainData.columns]

trainData.dropna(inplace = True)
trainData.reset_index(drop = True, inplace = True)

trainY = trainData['Survived']
trainX = trainData.drop(['Survived'], axis = 1)

# let's take a look at whether individual attributes correlate with a passenger's survival
# print(trainY)
# print(trainX[:, np.where(attrX == 'Male')].flatten())

for attr in trainX.columns:
	xCol = trainX[attr].values.flatten()
	try:
		print('Correlation of \"Survived\" and \"{}\": {}'.format( 
			attr,
			np.corrcoef(trainY, xCol)[0, 1]
		))

		slope, intercept, rSqu, p, stdErr = stat.linregress(trainY, col)
		plot.scatter(trainY, col)
		plot.xlabel('Survived')
		plot.ylabel(attr)
		plot.plot(col, intercept + slope * col, c = 'r')
		plot.xlim([0.0, 1.0])
		# plot.show()
	except Exception: 
		continue # embarked does not yet contain any numbers and will throw an error

print(trainX.columns)
for attr in ['Age', 'SibSp', 'Parch', 'Fare']:
	plot.hist(trainX[attr])
	plot.xlabel(attr)
	# plot.show()

def colToHotOne(X, colName, newNameArr):
	vals = X[colName].values

	oneHotFrame = pd.DataFrame(
		np.array([]),
		columns = newNameArr
	)

	for name in newNameArr:
		X.assign(
			# Weg finden, das innerhalb einer Loop zu implementieren
			inplace = True
		)

	return X

# change ('C', 'Q', 'S') to hot-one encoding
embarked = trainX['Embarked'].values.flatten()
print(len(embarked))

embarkedOneHot = pd.DataFrame(
	np.array([np.array([1, 0, 0] if port == 'C' else ([0, 1, 0] if port == 'Q' else [0, 0, 1])) for port in embarked]),
	columns = ['C', 'Q', 'S']
)

trainX = trainX.assign(
	C = embarkedOneHot['C'].values,
	S = embarkedOneHot['S'].values,
	Q = embarkedOneHot['Q'].values
)
trainX.drop(['Embarked'], inplace = True, axis = 1)

# now we will change the class (1, 2, 3) to one hot encoding, too
pClass = trainX['Pclass'].values.flatten()
classOneHot = pd.DataFrame(
	np.array([np.array([1, 0, 0] if passenger == 1 else ([0, 1, 0] if passenger == 2 else [0, 0, 1])) for passenger in pClass]),
	columns = ['class1', 'class2', 'class3']
)
trainX = trainX.assign(
	class1 = classOneHot['class1'].values,
	class2 = classOneHot['class2'].values,
	class3 = classOneHot['class3'].values
)

trainX.drop(['Pclass'], axis = 1, inplace = True)

print(trainX.head(20))


# Now, I standardize Age, SibSp, Parch and Fare
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

cols = ['Age', 'SibSp', 'Parch', 'Fare']

for col in cols:
	colValues = trainX[col].values.reshape(-1, 1)
	scaler.fit(colValues)
	trainX[col] = scaler.transform(colValues) 

# print(trainX.head(20))


# extract some data to validate from the training
def randTestData(Y, X, num):
	from random import randint
	drawn = []
	testY = []
	testX = []

	length = len(Y)
	i = randint(0, length - 1)

	while len(testY) < num:
		length = len(Y)

		while i in drawn:
			i = randint(0, length - 1)

		drawn.append(i)

		testY.append(Y[i])
		testX.append(X[i])

		Y = np.delete(Y, i)
		X = np.delete(X, i, 0)

	return (np.array(testY), np.array(testX), Y, X)

testValsY, testValsX, trainValsY, trainValsX = randTestData(trainY.values, trainX.values, int(0.1 * len(trainY)))

print('testValsY')
print(pd.DataFrame(testValsY).head(5), len(testValsY))

print('testValsX')
print(pd.DataFrame(testValsX).head(5), len(testValsX))

print('trainValsY')
print(pd.DataFrame(trainValsY).head(5), len(trainValsY))

print('trainValsX')
print(pd.DataFrame(trainValsX).head(5), len(trainValsX))

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(
	64,
	activation = 'relu',
	input_shape = (len(trainValsX[0]), )
))
model.add(Dense(8, activation = 'relu'))
model.add(Dropout(0.5))
# model.add(Dense(32, activation = 'relu'))
# model.add(Dropout(0.5))
# model.add(Dense(16, activation = 'relu'))
# model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(
	loss = 'binary_crossentropy',
	optimizer = 'rmsprop',
	metrics = ['accuracy']
)

print(model.summary())

classifier = model.fit(
	trainValsX, trainValsY,
	batch_size = 5,
	epochs = 128,
	verbose = 2,
	validation_data = (testValsX, testValsY)
)



def correctPreds(predY, testY):
	corrects = 0
	dataLen = len(predY)
	for i in range(dataLen - 1):
		if testY[i] == predY[i]:
			corrects += 1
	return corrects / dataLen

def applySVM(trainX, trainY, testX, testY, kernel):
	from sklearn import svm
	
	C = 1.0 # penalty parameter for error term
	classifier = svm.SVC(kernel = kernel, C = C).fit(trainX, trainY)

	predY =  classifier.predict(testX)

	print('Percentage of correctly predicted testYs using a Support Vector Machine with a ' + kernel + ' kernel: ' + str(correctPreds(predY, testY)) + '\n')

	return classifier

def logRegression(trainX, trainY, testX, testY):
	from sklearn.linear_model import LogisticRegression
	classifier = LogisticRegression().fit(trainX, trainY)
	predY = classifier.predict(testX)

	print('Percentage of correctly predicted testYs using logistic regression: ' + str(correctPreds(predY, testY)))

	return classifier

logClassifier = logRegression(trainValsX, trainValsY, testValsX, testValsY)

applySVM(trainValsX, trainValsY, testValsX, testValsY, 'linear') # linear kernel scores about as well as the decesion trees
applySVM(trainValsX, trainValsY, testValsX, testValsY, 'sigmoid') # about 50% correct predictions
applySVM(trainValsX, trainValsY, testValsX, testValsY, 'rbf')

