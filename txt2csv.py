import pandas as pd
import re

txt_file = "synthetic_data_dp.txt"
with open(txt_file, "r", encoding="utf-8") as file:
    lines = file.readlines()

data_list = []
for line in lines:
    pairs = re.findall(r"(\w[\w\s-]+): ([^,]+)", line)
    data_dict = {key.strip(): value.strip() for key, value in pairs}
    data_list.append(data_dict)

df = pd.DataFrame(data_list)
rename_dict = {
    "Race": "race",
    "Sex": "sex",
    "Age": "age",
    "Workclass": "workclass",
    "fnlwgt": "fnlwgt",
    "Education": "education",
    "Educated Years": "education-num",
    "Marital Status": "marital-status",
    "Relationship": "relationship",
    "Occupation": "occupation",
    "Income": "capital-gain",  # 原始 "Income" 对应 "capital-gain"
    "Loss": "capital-loss",    # 原始 "Loss" 对应 "capital-loss"
    "Work Hours": "hours-per-week",
    "Native Country": "native-country",
    "Income Level": "income"
}
df.drop(columns= 'ID', inplace=True)
df.rename(columns=rename_dict, inplace=True)
new_column_order = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

df = df[new_column_order]
df.to_csv("./data/generated_adult100_dp.csv", index=False)

print("CSV 文件已成功创建并保存为 'output.csv'")
