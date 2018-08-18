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


# Now, I standardize Age, SibSp, Parch and Fare
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()

# normCols = ['Age', 'SibSp', 'Parch', 'Fare']
# print(trainX[normCols].describe())
# toBeNormed = trainX[normCols].values
# scaler.fit(toBeNormed)
# norm = pd.DataFrame(scaler.transform(toBeNormed), columns = normCols)

# print(norm.describe())

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

testY, testX, trainY, trainX = randTestData(trainY, trainX, int(0.1 * len(trainY)))



# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout

# model = Sequential()

# model.add(Dense(
# 	64, 
# 	activation = 'relu', 
# 	input_shape = (len(trainX[0]), )
# ))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation = 'relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation = 'sigmoid'))

# model.compile(
# 	loss = 'binary_crossentropy',
# 	optimizer = 'rmsprop',
# 	metrics = ['accuracy']
# )
