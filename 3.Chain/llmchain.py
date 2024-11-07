from langchain_community.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv,find_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

_=load_dotenv(find_dotenv())

llm = ChatOpenAI(temperature=0,model="gpt-3.5-turbo",openai_api_key=os.getenv("OPENAI_API_KEY"))

prompt_template=PromptTemplate.from_template("What is the capital of {country} and how many countries are there  in  the {continent}")

chain=LLMChain(llm=llm,prompt=prompt_template)

country="India"
continent="Asia"


print(chain.invoke({"country":country,"continent":continent}))
