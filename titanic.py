import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import scipy.stats as stat

attributeNames = [
	'survival',	# 0 = No, 1 = Yes, to be predicted
	'pclass',	# ticket class; 1 = 1st, 2 = 2nd, 3 = 3rd
	'sex',
	'age',
	'sibsp',	#  number of siblings / spouses aboard
	'parch',	# number of parents / children aboard
	'ticket',	# ticket number
	'fare',		# price the passenger was charged with
	'cabin',	# Cabin number
	'embarked' 	# Port of Embarkation: C = Cherbourg, Q = Queenstown, S = Southampton
]

trainData = pd.read_csv(
	'./data/train.csv', 
	na_values = ['NaN']
)

# print(trainData.describe())
# print(trainData.head(15))
trainData = trainData.drop(['Name', 'Ticket', 'Cabin'], axis = 1)
trainData.replace(('male', 'female'), (1, 0), inplace = True)
trainData.replace(('C', 'Q', 'S'), (0, 1, 2), inplace = True)
# print(trainData.head(15))

# print(trainData.columns)
trainData.columns = [attr if attr != 'Sex' else 'Male' for attr in trainData.columns]
trainData = trainData.dropna()
# print(trainData.describe())

trainY = trainData['Survived'].values
trainX = trainData.drop(['Survived'], axis = 1)#.values
attrX = trainX.columns # for working with columnsindex = np.where(arr == 'colName')
trainX = trainX.values

# print(trainY.shape, trainX.shape, len(attrX) == len(trainX[0]))

# let's take a look whether individual attributes correlate with a passenger's survival
print(trainY)
print(trainX[:, np.where(attrX == 'Male')].flatten())

for attr in attrX:
	dta = trainX[:, np.where(attrX == attr)].flatten()
	print(
		'Correlation of \"Survived\" and \"{}\": {}'.format( 
			attr,
			np.corrcoef(trainY, dta)[0, 1]
		)
	)

	slope, intercept, rSqu, p, stdErr = stat.linregress(trainY, dta)
	def f(x):
		return intercept + slope * x


	plot.scatter(trainY, dta)
	plot.xlabel('Survived')
	plot.ylabel(attr)
	plot.plot(dta, f(dta), c = 'r')
	plot.xlim([0.0, 1.0])
	plot.show()

