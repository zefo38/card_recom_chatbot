import langchain
import transformers
from CsvLoader import csvload
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
import langchain_core
import langchain_text_splitters
from langchain.agents import create_pandas_dataframe_agent
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.agents import AgentType


model_id = 'MLP-KTLim/llama-3-Korean-Bllossom-8B'
path = './consumer_data.csv'
encoding = 'utf-8'
source_column = '고객번호'
c = csvload(path, encoding, source_column)
data = c.get_csv()

class usingagent():
    def __init__(self):
        self.temperature = 0.0
        self.model = model_id
        self.data = data
        self.verbose = False

    def getagent(self, temperature, model, verbose, data):
        agent = create_pandas_dataframe_agent(
            HuggingFacePipeline(temperature = self.temperature, model = self.model),
            data,
            verbose = self.verbose,
        )
        return agent

    def 