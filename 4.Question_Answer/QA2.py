from langchain_community.document_loaders import CSVLoader
import os
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

llm=ChatOpenAI(temperature=0,model="gpt-3.5-turbo-instruct",openai_api_key=os.getenv("OPENAI_API_KEY"))

loader = CSVLoader(file_path="customers-50.csv")
docs=loader.load()
embeddings = OpenAIEmbeddings()
embed=embeddings.embed_query("Hii Makkale!")
print(len(embed))
db=DocArrayInMemorySearch.from_documents(
    docs,
    embeddings
)

query="What is the name of the customer with customer_id '8C2811a503C7c5a'?"

docs=db.similarity_search(query,k=50)

print(len(docs))

retriver=db.as_retriever()


q_docs="\n".join([docs[i].page_content for i in range(len(docs))])

print(q_docs)

from langchain.indexes import VectorstoreIndexCreator

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embeddings,
).from_loaders([loader])

response=llm.call_as_llm(f"{q_docs} Question:please list out\
                          all the customer names ")
print(response)

qa_stuff=RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=retriver,verbose=True)

query="please list out all the customer names "
response=qa_stuff.run(query)

print("\n\n",response,"\n\n")


response=index.query(query,llm=llm)

print(response)


