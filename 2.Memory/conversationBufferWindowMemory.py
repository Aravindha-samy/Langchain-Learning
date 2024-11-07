import langchain_community.chat_models as chat
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv,find_dotenv
_=load_dotenv(find_dotenv())
import warnings
warnings.filterwarnings("ignore")

chat=ChatOpenAI(temperature=0,api_key=os.getenv("OPENAI_API_KEY"),model="gpt-3.5-turbo")

memory=ConversationBufferWindowMemory(k=2)

conversation=ConversationChain(llm=chat,memory=memory,verbose=True)

print("Enter 'exit' to quit")
while True:
    user_input=input("You: ")   
    if user_input.lower()=="exit":
        break
    response=conversation.predict(input=user_input)
    print(response)
print("--------------------------------")
print(memory.buffer)
print("--------------------------------")
print(memory.load_memory_variables({}))
