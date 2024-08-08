from CSV2TXT import csv2txt
import pandas as pd

save_path = './customer_txt_file/customer_txt_data.txt'

data = pd.read_csv('./processed_file1.csv')

c2t = csv2txt(data, save_path)

c2t.transform()