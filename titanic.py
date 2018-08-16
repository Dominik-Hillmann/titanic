import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import scipy.stats as stat

trainData = pd.read_csv(
	'./data/train.csv', 
	na_values = ['']
)
testData = pd.read_csv(
	'./data/test.csv',
	na_values = ['']
)

trainData = trainData.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis = 1)
# I drop these attributes since I think that either:
#	- they don't contain information useful to know whether someone survived (Name, Passenger ID)
#	- I don't have the means to dissern useful information from it (all three, Cabin and Ticket because maybe they contain information about where cabin was within the ship

trainData.replace(('male', 'female'), (1, 0), inplace = True)
trainData.columns = [attr if attr != 'Sex' else 'Male' for attr in trainData.columns]

trainData.dropna(inplace = True)
trainData.reset_index(drop = True, inplace = True)

trainY = trainData['Survived']
trainX = trainData.drop(['Survived'], axis = 1)

# let's take a look whether individual attributes correlate with a passenger's survival
# print(trainY)
# print(trainX[:, np.where(attrX == 'Male')].flatten())

for attr in trainX.columns:
	xCol = trainX[attr].values.flatten()
	try:
		print('Correlation of \"Survived\" and \"{}\": {}'.format( 
			attr,
			np.corrcoef(trainY, xCol)[0, 1]
		))
	except Exception: 
		continue # embarked does not yet contain any numbers and will throw an error

	# slope, intercept, rSqu, p, stdErr = stat.linregress(trainY, col)

	# plot.scatter(trainY, col)
	# plot.xlabel('Survived')
	# plot.ylabel(attr)
	# plot.plot(col, intercept + slope * col, c = 'r')
	# plot.xlim([0.0, 1.0])
	# plot.show()

def colToHotOne(X, colName, newNameArr):

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
embarkedOneHot = pd.DataFrame(
	np.array([np.array([1, 0, 0] if passenger == 1 else ([0, 1, 0] if passenger == 2 else [0, 0, 1])) for passenger in pClass]),
	columns = ['class1', 'class2', 'class3']
)
trainX = trainX.assign(
	class1 = embarkedOneHot['class1'].values,
	class2 = embarkedOneHot['class2'].values,
	class3 = embarkedOneHot['class3'].values
)
trainX.drop(['Pclass'], inplace = True, axis = 1)
print(trainX.head(20))


# print(trainX.head(50))
# # Problem ist hier gefunden:
# # 	embarkedOneHot hat fortlaufende Indizes von 1 bis 712,
# # 	waehrend trainData fehlende Indizes besitzt, weil Punkte durch .dropna() geloescht worden sind
# # 	entweder muss embarkedOneHot wie trainData indiziert werden oder vice versa
# # 	vermutlich leichter durch Methode trainData neu fortlaufend zu indexieren
# # 	danach sollte obige Operation ohne Entstehung von NaNs funktionieren.

# for attr in trainData.columns:
# 	print(len(trainData[[attr]]))

# for attr in embarkedOneHot.columns:
# 	trainData[[attr]] = embarkedOneHot[[attr]]
# for attr in trainOld.columns:
# 	trainData[[attr]] = trainOld[[attr]]
# # trainData = trainData.dropna()

# # for attr in trainData.columns:
# # 	print(len(trainData[[attr]]))
# # trainData = trainData.add(embarkedOneHot[['Queenstown']], axis = 'columns')
# # print(trainData.head(40))
# # print(trainOld.head(40))
# print(len(trainData[['Male']]), len(trainData[['Queenstown']]))
# # trainData = pd.concat(trainData, embarkedOneHot, axis = 1)
# # print(trainData.head(30))
# # trainData = trainData.drop(['Embarked'])