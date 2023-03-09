# Alaa Mohamed Galal
# 19P4206

import pandas as pd
from apyori import apriori

# Store the data from the csv file
Store_Data = pd.read_csv("store_data.csv", header=None)

# Convert the dataframe into a list
DataSet = []
for i in range(0,7501):
    record = []
    for j in range(0,20):
        value = Store_Data.values[i,j]
        if str(value) != 'nan':
            record.append(value)
        DataSet.append(record)
DataSet_List = Store_Data.values.tolist()
print(DataSet_List)

# Remove "nan" values from dataframe
Updated_List = [[x for x in sublist if str(x) != 'nan'] for sublist in DataSet_List]
print(Updated_List)

# Apply apriori on the dataframe
Association_Rule = apriori(DataSet, min_support=0.005, min_confidence=0.2, min_length=1)
Association_Result = list(Association_Rule)
print("There are {} relation derived.".format(len(Association_Result)))

# Convert the output into the rule format
for item in Association_Result:
    print(item)
    pair = item[0]
    items = [x for x in pair]
    print("Frequent item set: " + str(items))
    print("Support: " + str(item[1]))
    if (len(pair) > 1):
        for rule in item[2]:
            print("Rule: " + str(rule[0]) + " -> " + str(rule[1]))
            print("Confidence: " + str(rule[2]))
            print("Lift: " + str(rule[3]))
            print("----------------------")