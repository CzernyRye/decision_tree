# CSE6242/CX4242 Homework 4 Pseudocode
# You can use this skeleton code for Task 1 of Homework 4.
# You don't have to use this code. You can implement your own code from scratch if you want.
from __future__ import division
from math import log
import csv

"""
# Attribute description (Global variable)
"""
workclass = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
             "Local-gov", "State-gov", "Without-pay", "Never-worked"]
education = ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
             "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters",
             "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"]
marital_status = ["Married-civ-spouse", "Divorced", "Never-married", "Separated",
                  "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
occupation = ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
              "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
              "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"]
relationship = ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"]
race = ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
sex = ['Female', 'Male']
native_country = ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
                  "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China",
                  "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam",
                  "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador",
                  "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand",
                  "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"]

# Here I store the description of each attribute
# dict = {attribute: [ values,  column_index_in_data ]}
attribute_property = {'age':{'val':'continuous','col':0}, 'workclass':{'val':workclass,'col':1},'fnlwgt':{'val':'continuous','col':2},'education':{'val':education,'col':3},
				 'education-num':{'val':'continuous','col':4},'marital-status':{'val':marital_status,'col':5},'occupation':{'val':occupation,'col':6},
				 'relationship':{'val':relationship,'col':7},'race':{'val':race,'col':8},'sex':{'val':sex,'col':9},'capital-gain':{'val':'continuous','col':10},
				 'capital-loss':{'val':'continuous','col':11},'hours-per-week':{'val':'continuous','col':12},'native-country':{'val':native_country,'col':13}}
"""
# END
"""

# Implement your decision tree below
class DecisionTree():

	tree = {}

	def learn(self, data, attributes):
		'''''''''
		Returns a new decision tree based on the example given
		'''''''''
		data = data[:]
		vals = [record[-1] for record in data]
		default = majority_value(data)

		# If the dataset is empty or the attributes list is empty, return the
		# default value.
		if not data or (len(attributes)) <= 0:
			return default
		#  If all the records in the dataset have the same classification,
		# return that classification
		elif vals.count(vals[0]) == len(vals):
			return vals[0]

		else:
			# Choose the next best attribute to best classify our data
			best_params = choose_attribute(data, attributes)
			best = best_params[0]
			flag = best_params[1]

			# Create a new decision tree/node with the best attribute and an empty
			# dictionary object--we'll fill that up next.
			tree = {best:{}}

			# Create a new decision tree/ sub-node for each of the values
			# in the best attribute field
			if flag == False:
				for val in get_values(data,best):
					subtree = self.learn(get_example(data,best,val),
										 [attr for attr in attributes if attr != best])
					tree[best][val] = subtree
			elif flag == True:
				# the data is continuous,  we'll split the data to left and right part
				col = attribute_property[best]['col']
				split = best_params[2]
				left_data = [record for record in data if float(record[col]) < split]
				right_data = [record for record in data if float(record[col]) > split]
				left_subtree = self.learn(left_data,
										 [attr for attr in attributes if attr != best])
				val = '<' + str(split)
				tree[best][val] = left_subtree
				right_subtree = self.learn(right_data,
										  [attr for attr in attributes if attr != best])
				val = '>' + str(split)
				tree[best][val] = right_subtree

		self.tree = tree
		return self.tree



	def classify(self, test_instance):

		# dict = self.tree
		# attr = dict.keys()[0]
		# while attribute_property.has_key(attr):
		# 	attr = dict.keys()[0]
		# 	col = attribute_property[attr]['col']
		# 	val = test_instance[col]
		# 	dict = dict[attr]
		# 	dict = dict[val]
		# 	if type(dict) != type("string"):
		# 		attr = dict.keys()[0]
		# 	else:
		# 		break
        #
		# result = dict # baseline: always classifies as <=50K

		result = "<=50K"

		return result


def get_example(data,best,val):
	col = attribute_property[best]['col']
	example = [record for record in data if record[col] == val]
	return example

def get_values(data,best):
	values = attribute_property[best]['val']
	return values

def majority_value(data):
	val_freq = {}
	if len(data) != 0:
		for record in data:
			if val_freq.has_key(record[-1]):
				val_freq[record[-1]] += 1.0
			else:
				val_freq[record[-1]] = 1.0
		label = max(val_freq.iterkeys(), key=(lambda key: val_freq[key]))
	else:
		label = "<=50K"
	return label

def entropy(data):
	# Calcuate the entropy of given data for the target attribute
	val_freq = {}
	data_entropy = 0.0

	for record in data:
		if val_freq.has_key(record[-1]):
			val_freq[record[-1]] += 1.0
		else:
			val_freq[record[-1]] = 1.0
	for freq in val_freq.values():
		data_entropy += (-freq/len(data)) * log(freq/len(data),2)

	return data_entropy

def gain(data,attr):
	# Calculates the information gain (reduction in entropy) that would
	#  result by splitting the data on the chosen attribute (attr).
	col = attribute_property[attr]['col']
	val_freq = {}
	subset_entropy = 0.0

	# Calculate the frequency of each of the values in the target attribute
	for record in data:
		if val_freq.has_key(record[col]):
			val_freq[record[col]] += 1.0
		else:
			val_freq[record[col]] = 1.0
	# Calculate the sum of the entropy for each subset of records weighted
	# by their probability of occuring in the training set.
	for val in val_freq:
		val_prob = val_freq[val]/sum(val_freq.values())
		data_subset = [record for record in data if record[col] == val]
		subset_entropy += val_prob * entropy(data_subset)

	# Substract the entropy of the chosen attribute from the whole dataset
	# with respect to target attribute and return it
	return entropy(data) - subset_entropy

def choose_best_split(data,attr):
	col = attribute_property[attr]['col']
	vals = []
	splits = []
	splits_gain = {}
	seen_val = set()

	# Sort the values in data with respect to the attribute
	for record in data:
		if record[col] not in seen_val:
			vals.append(int(record[col]))
			vals.sort()
			seen_val.add(record[col])

	# Compute the splits for each pair of adjacant sorted values,
	# then calculate the infomation gain with respect to split
	for idx,val in enumerate(vals):
		tmp1 = val
		if idx > 0:
			splits.append((tmp2 + tmp1)/2)
		tmp2 = tmp1

	for split in splits:
		# Calculate informatino gain of split with respect to target attribute
		left_freq = 0.0
		right_freq = 0.0
		left_subset = []
		right_subset = []

		for record in data:
			if float(record[col]) < split:
				left_freq += 1.0
				left_subset.append(record)
			else:
				right_freq += 1.0
				right_subset.append(record)

		subset_entropy = (left_freq)/(left_freq+right_freq) * entropy(left_subset) + (right_freq)/(left_freq+right_freq) * entropy(right_subset)
		splits_gain[split] = entropy(data) - subset_entropy
		if len(data) == 0:
			print 'best split is empty'
			return
    	else:
			best_split = max(splits_gain.iterkeys(), key=(lambda key: splits_gain[key]))
			return splits_gain[best_split],best_split

def choose_attribute(data,attributes):
	attr_gain = {}
	flag = False # The flag states the best attribute which has  the max information gain  is continuous or not
	# Evaluate the information gain with respect to attribute
	for attr in attributes:
		# Check the attribute (attr) is categorial or continous
		if attribute_property[attr]['val'] != 'continuous':
			attr_gain[attr] = gain(data, attr)
		else:
		# The attribute is continous, we need to find the split in this attribute
		# which it give the maximum infomation gain
			split_params = choose_best_split(data, attr)
			if split_params is not None:
				attr_gain[attr] = split_params[0]

	# Return the attribute ( best ) gives the most information gain with respect to target attribute
	# ; the flag states the attribute is contiuous or not; and if it is contiuous, return the split point as well
	best = max(attr_gain.iterkeys(), key=(lambda key: attr_gain[key]))
	if attribute_property[best]['val'] == 'continuous':
		flag = True
		return best, flag, split_params[1] # attribute, flag, split point
	else:
		return best, flag

def run_decision_tree():
	
	# Load data set
	with open("./data/adult_nona.tsv") as tsv:
		data = [tuple(line) for line in csv.reader(tsv, delimiter="\t")]
	print "Number of records: %d" % len(data)


	# Split training/test sets
	# You need to modify the following code for cross validation.
	K = 10
	training_set = [x for i, x in enumerate(data) if i % K != 9]
	test_set = [x for i, x in enumerate(data) if i % K == 9]

	
	tree = DecisionTree()
	# Construct a tree using training set
	attributes = [attr for attr in attribute_property.keys() if (attr != 'fnlwgt') and (attr != 'capital-gain') and
				  (attr != 'capital-loss') and (attr != 'hours-per-week') and (attr != 'education-num') and (attr != 'age')]
	tree.learn( training_set, attributes)

	# Classify the test set using the tree we just constructed
	results = []
	for instance in test_set:
		result = tree.classify( instance[:-1] )
		results.append( result == instance[-1] )

	# Accuracy
	accuracy = float(results.count(True))/float(len(results))
	print "accuracy: %.4f" % accuracy
	
	
	# Writing results to a file (DO NOT CHANGE)
	f = open("result.txt", "w")
	f.write("accuracy: %.4f" % accuracy)
	f.close()


if __name__ == "__main__":
	run_decision_tree()