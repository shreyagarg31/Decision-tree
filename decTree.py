import numpy as np
import pandas as pd
eps = np.finfo(float).eps
from numpy import log2 as log
from itertools import dropwhile, takewhile
import pprint
import csv

dataset = {'age':['youth','youth','middle_aged','senior','senior','senior','middle_aged','youth','youth','senior','youth','middle_aged','middle_aged','senior'],
'income':['high','high','high','medium','low','low','low','medium','low','medium','medium','medium','high','medium'],
'student':['no','no','no','no','yes','yes','yes','no','yes','yes','yes','no','yes','no'],
'credit_rating':['fair','excellent','fair','fair','fair','excellent','excellent','fair','fair','fair','excellent','excellent','fair','excellent'],
'buys_computer':['no','no','yes','yes','yes','no','yes','no','yes','yes','yes','yes','yes','no']}

df = pd.DataFrame(dataset,columns=['age','income','student','credit_rating','buys_computer'])

def find_entropy(df):
    Class = df.keys()[-1]   #To make the code generic, changing target variable class name
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value]/len(df[Class])
        entropy += -fraction*np.log2(fraction)
    return entropy
  
  
def find_entropy_attribute(df,attribute):
  Class = df.keys()[-1]   #To make the code generic, changing target variable class name
  target_variables = df[Class].unique()  #This gives all 'Yes' and 'No'
  variables = df[attribute].unique()    #This gives different features in that attribute (like 'Hot','Cold' in Temperature)
  entropy2 = 0
  for variable in variables:
      entropy = 0
      for target_variable in target_variables:
          num = len(df[attribute][df[attribute]==variable][df[Class] ==target_variable])
          den = len(df[attribute][df[attribute]==variable])
          fraction = num/(den+eps)
          entropy += -fraction*log(fraction+eps)
      fraction2 = den/len(df)
      entropy2 += -fraction2*entropy
  return abs(entropy2)


def find_winner(df):
    Entropy_att = []
    IG = []
    for key in df.keys()[:-1]:
#         Entropy_att.append(find_entropy_attribute(df,key))
        IG.append(find_entropy(df)-find_entropy_attribute(df,key))
#    print(IG)
    return df.keys()[:-1][np.argmax(IG)]
  
  
def get_subtable(df, node,value):
  return df[df[node] == value].reset_index(drop=True)


def buildTree(df,tree=None): 
    Class = df.keys()[-1]   #To make the code generic, changing target variable class name
    
    #Here we build our decision tree

    #Get attribute with maximum information gain
    node = find_winner(df)
    print(node)
    
    #Get distinct value of that attribute e.g Salary is node and Low,Med and High are values
    attValue = np.unique(df[node])
    
    #Create an empty dictionary to create tree    
    if tree is None:                    
        tree={}
        tree[node] = {}
    
   #We make loop to construct a tree by calling this function recursively. 
    #In this we check if the subset is pure and stops if it is pure. 

    for value in attValue:
        
        subtable = get_subtable(df,node,value)
        clValue,counts = np.unique(subtable['buys_computer'],return_counts=True)                        
        
        if len(counts)==1:#Checking purity of subset
            tree[node][value] = clValue[0]                                                    
        else:        
            tree[node][value] = buildTree(subtable) #Calling the function recursively 
                   
    return tree

def predict(inst,tree):
    #This function is used to predict for any input variable 
    
    #Recursively we go through the tree that we built earlier

    for nodes in tree.keys():        
        
        value = inst[nodes]
        tree = tree[nodes][value]
        prediction = 0
            
        if type(tree) is dict:
            prediction = predict(inst, tree)
        else:
            prediction = tree
            break;                            
        
    return prediction

def test_predictions(tree,df):
	num_data = 0
	num_correct = 0
	for index,row in df.items():
		r1 = pd.Series(row)
		prediction = predict(r1,tree)
		for key in row:
			if key=='buys_computer':
				val = row[key]

		if prediction == val:
			num_correct += 1
		num_data += 1
	return round(num_correct/num_data, 2)

tree=buildTree(df)
pprint.pprint(tree)


df_test = pd.read_csv('testData.txt',sep='\s+').to_dict(orient= 'index')

print("Accuracy of test data:")
print(str(test_predictions(tree, df_test)*100.0) + '%')

data1 = {'age':'senior','income':'high','student':'yes','credit_rating':'fair'}
data2 = {'age':'youth','income':'high','student':'yes','credit_rating':'fair'}
data3 = {'age':'middle_aged','income':'low','student':'no','credit_rating':'excellent'}

inst1 = pd.Series(data1)
inst2 = pd.Series(data2)
inst3 = pd.Series(data3)

prediction1 = predict(inst1,tree)
prediction2 = predict(inst2,tree)
prediction3 = predict(inst3,tree)

print("Prediction values for the samples are:")
print(prediction1)
print(prediction2)
print(prediction3)