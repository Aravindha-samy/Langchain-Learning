from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain,SimpleSequentialChain
from dotenv import load_dotenv,find_dotenv
import os

_=load_dotenv(find_dotenv())

llm=ChatOpenAI(temperature=0,model="gpt-3.5-turbo",openai_api_key=os.getenv("OPENAI_API_KEY"))

prompt_template_1=PromptTemplate.from_template("What is the capital of {country}?")
chain_1=LLMChain(llm=llm,prompt=prompt_template_1)
prompt_template_2=PromptTemplate.from_template("What is the population of {country}?")

chain_2=LLMChain(llm=llm,prompt=prompt_template_2)

overall_chain=SimpleSequentialChain(chains=[chain_1,chain_2],verbose=True)

country="India"

print(overall_chain.invoke(country))


