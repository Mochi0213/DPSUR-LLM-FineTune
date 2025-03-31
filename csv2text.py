import pandas as pd
import random

random.seed(213)
def generate_random_id():
    return str(random.randint(1000, 9999))
def row_to_text(row):
    return f"ID: {row['ID']}, Race: {row['race']}, Sex: {row['sex']}, Age: {row['age']}, Workclass: {row['workclass']}, fnlwgt: {row['fnlwgt']}, Education: {row['education']}, Educated Years: {row['education-num']}, "\
           f"Marital Status: {row['marital-status']}, Relationship: {row['relationship']}, Occupation: {row['occupation']}, Income: {row['capital-gain']}, Loss: {row['capital-loss']}, "\
           f"Work Hours: {row['hours-per-week']}, Native Country: {row['native-country']}, Income Level: {row['income']}"

df = pd.read_csv("./data/adult100.csv")
df["ID"] = [generate_random_id() for _ in range(len(df))]


text_data = df.apply(row_to_text, axis=1).tolist()

output_file = "processed_data.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for line in text_data:
        f.write(line + "\n")