import langchain
import langchain_core
import langchain_text_splitters
from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path = 'consumer_data.csv', encoding = 'utf-8' ,source_column = '고객번호', csv_args = {'delimiter' : '\n',})
data = loader.load()

print(data)