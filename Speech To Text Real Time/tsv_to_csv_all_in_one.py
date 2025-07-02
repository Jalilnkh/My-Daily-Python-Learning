import pandas as pd
from utils import convert_tsv_to_csv

first_in_path = "/home/jalilnkh/my_projects/My-Daily-Python-Learning/Speech To Text Real Time/az/train.tsv"
second_in_path = "/home/jalilnkh/my_projects/My-Daily-Python-Learning/Speech To Text Real Time/az/test.tsv"
third_in_path = "/home/jalilnkh/my_projects/My-Daily-Python-Learning/Speech To Text Real Time/az/dev.tsv"
fourth_in_path = "/home/jalilnkh/my_projects/My-Daily-Python-Learning/Speech To Text Real Time/az/validated.tsv"
fifth_in_path = "/home/jalilnkh/my_projects/My-Daily-Python-Learning/Speech To Text Real Time/az/other.tsv"

output_path = "/home/jalilnkh/my_projects/My-Daily-Python-Learning/Speech To Text Real Time/dataset/common_voices_az_azb_sentences.csv"
sen1 = convert_tsv_to_csv(first_in_path, ["path", "sentence"])
sen2 = convert_tsv_to_csv(second_in_path, ["path", "sentence"])
sen3 = convert_tsv_to_csv(third_in_path, ["path", "sentence"])
sen4 = convert_tsv_to_csv(fourth_in_path, ["path", "sentence"])
sen5 = convert_tsv_to_csv(fourth_in_path, ["path", "sentence"])

# Save as CSV
all_sen = pd.concat([sen1, sen2, sen3, sen4, sen5], ignore_index=True)
all_sen.to_csv(output_path, index=False, encoding="utf-8-sig")
