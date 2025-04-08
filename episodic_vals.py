import pandas as pd

# Load the dataset (adjust the separator if needed; here we assume tab-delimited data)
df = pd.read_csv("YA_GroupData.txt", sep="\t")
YA_DATA = df

# Drop any rows where 'ChosenObj' is missing, as those trials don't reveal a value.
df_clean = df.dropna(subset=["ChosenObj"])

# Convert 'ChosenObj' to a string (or integer) to use as a key
df_clean["ChosenObj"] = df_clean["ChosenObj"].astype(int).astype(str)

# Ensure the Pay column is numeric
df_clean["Pay"] = pd.to_numeric(df_clean["Pay"], errors="coerce")*100

# Group by 'ChosenObj' and calculate the mode of the 'Pay' values.
# Since each object is assigned a fixed point value per participant,
# the mode should reflect that assigned episodic value.
global_mapping = df_clean.groupby("ChosenObj")["Pay"].agg(lambda x: x.mode().iloc[0])

# Convert the Series to a dictionary
EPISODIC_VALUES_MAP = global_mapping.to_dict()

print("Global Episodic Values Map:")
for obj, value in EPISODIC_VALUES_MAP.items():
    print(f"Object {obj}: {value}")
num_objects = len(EPISODIC_VALUES_MAP)
print(f"Number of objects in the episodic values map: {num_objects}")
