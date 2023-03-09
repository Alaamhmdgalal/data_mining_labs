import pandas as pd
from apyori import apriori

# Store the data from the csv file without header
Weather_Nominal = pd.read_csv("weather_nominal.csv", skiprows=0)
# Check the data types
print(Weather_Nominal.dtypes)
# Convert the non object data into objects
Weather_Nominal['id'] = Weather_Nominal['id'].apply(str)
Weather_Nominal['windy'] = Weather_Nominal['windy'].apply(str)
# Check the types after being updated
print(Weather_Nominal.dtypes)
print(Weather_Nominal)

# Convert the dataframe into a list
Reader = []
for i in range(0,13):
    record = []
    for j in range(0, 6):
        value = Weather_Nominal.values[i, j]
        if str(value) != 'nan':
            record.append(value)
        Reader.append(record)
DataSet_List = Weather_Nominal.values.tolist()
print(DataSet_List)

# Apply apriori on the dataframe
Association_Rule = apriori(Reader, min_support=0.1, min_confidence=0.7, min_length=1)
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