import langchain
from CsvLoader import csvload
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
import langchain_core
import langchain_text_splitters
from langchain.agents import create_pandas_dataframe_agent
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.agents import AgentType


model_id = 'jayiuk/basic_model'
path = './consumer_data.csv'
encoding = 'utf-8'
source_column = '고객번호'
c = csvload(path, encoding, source_column)
data = c.get_csv()

class usingagent():
    def __init__(self, temperature, model, verbose):
        self.temperature = temperature
        self.model = model
        self.verbose = verbose

    def getagent(self):
        agent = create_pandas_dataframe_agent()