import os
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.llms import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

file = 'customers-50.csv'
loader = CSVLoader(file_path=file)
from langchain.indexes import VectorstoreIndexCreator

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=OpenAIEmbeddings()
).from_loaders([loader])


query = "What is the name of the customer with customer_id '8C2811a503C7c5a'?"

llm=OpenAI(temperature=0,model="gpt-3.5-turbo-instruct",openai_api_key=os.getenv("OPENAI_API_KEY"))

response = index.query(query, llm=llm)

print(response)
