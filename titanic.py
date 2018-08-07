import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

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
	na_values = ['NaN'],
)

# print(trainData.describe())
print(trainData.head(15))
trainData = trainData.drop(['Name', 'Ticket', 'Cabin'], axis = 1)
trainData.replace(('male', 'female'), (1, 0), inplace = True)
trainData.replace(('C', 'Q', 'S'), (0, 1, 2), inplace = True)
print(trainData.head(15))

cols = trainData.columns
print(cols)
cols = [x if x % 2 else None for x in items]