import pandas as pd

"""Preparing data"""

OA_df = pd.read_csv("OA_GroupData.txt", sep="\t")
OA_DATA = OA_df

df = pd.read_csv("Norming_Data.csv", dtype={"stimulusitem2": str})

# Find the average of two median senmantic values
df['semantic_value'] = (df['med1d'] + df['med2d']) / 2

# Determine range for normalization
min_val = df['semantic_value'].min()
max_val = df['semantic_value'].max()

# Normalize values
df['nacs_activation'] = (df['semantic_value'] - min_val) / (max_val - min_val)

df['nacs_scaled'] = round(df['nacs_activation'] * 100)


# Convert to dictionary: object ID (as string) → activation value
SEM_ACTIVATION_MAP = {
    str(row['stimulusitem2']): row['nacs_scaled']
    for _, row in df.iterrows()
}
print("Global Semantic Values Map:")
for obj, value in SEM_ACTIVATION_MAP.items():
    print(f"Object {obj}: {value}")
num_objects = len(SEM_ACTIVATION_MAP)
print(f"Number of objects in the semantic values map: {num_objects}")


