import warnings
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv,find_dotenv
_=load_dotenv(find_dotenv())

from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory

chat=ChatOpenAI(temperature=0,api_key=os.getenv("OPENAI_API_KEY"),model="gpt-3.5-turbo")

memory=ConversationSummaryMemory(llm=chat)  

conversation=ConversationChain(llm=chat,memory=memory,verbose=True)

print("Enter 'exit' to quit")
while True:
    user_input=input("You: ")

    if user_input.lower()=="exit":
        break
    else:
        print(conversation.invoke({"input":user_input}))

print(memory.load_memory_variables({}))
print("--------------------------------")
print(memory.buffer)
print("--------------------------------")
