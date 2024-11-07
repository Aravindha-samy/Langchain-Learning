from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationTokenBufferMemory
import os
from dotenv import load_dotenv,find_dotenv
_=load_dotenv(find_dotenv())
import warnings
warnings.filterwarnings("ignore")

chat=ChatOpenAI(temperature=0,api_key=os.getenv("OPENAI_API_KEY"),model="gpt-3.5-turbo")

memory=ConversationTokenBufferMemory(llm=chat,max_token=1)

memory.save_context({"input": "Backpropagation is what?"},
                    {"output": "Beautiful!"})
memory.save_context({"input": "Chatbots are what?"}, 
                    {"output": "Charming!"})
memory.save_context({"input": "AI is what?!"},
                    {"output": "Amazing!"})
memory.save_context({"input": "Backpropagation is what?"},
                    {"output": "Beautiful!"})
memory.save_context({"input": "Chatbots are what?"}, 
                    {"output": "Charming!"})
memory.save_context({"input": "AI is what?!"},
                    {"output": "Amazing!"})
memory.save_context({"input": "Backpropagation is what?"},
                    {"output": "Beautiful!"})
memory.save_context({"input": "Chatbots are what?"}, 
                    {"output": "Charming!"})

# conversation=ConversationChain(llm=chat,memory=memory,verbose=True) 

# print("Enter 'exit' to quit")
# while True:
#     user_input=input("You: ")   

#     if user_input.lower()=="exit":
#         break
#     response=conversation.predict(input=user_input)
#     print(response)
memory.save_context({"input": "AI is what?!"},
                    {"output": "Amazing!"})
memory.save_context({"input": "Backpropagation is what?"},
                    {"output": "Beautiful!"})
memory.save_context({"input": "Chatbots are what?"}, 
                    {"output": "Charming!"})
memory.save_context({"input": "AI is what?!"},
                    {"output": "Amazing!"})
memory.save_context({"input": "Backpropagation is what?"},
                    {"output": "Beautiful!"})
memory.save_context({"input": "Chatbots are what?"}, 
                    {"output": "Charming!"})
memory.save_context({"input": "AI is what?!"},
                    {"output": "Amazing!"})
memory.save_context({"input": "Backpropagation is what?"},
                    {"output": "Beautiful!"})
memory.save_context({"input": "Chatbots are what?"}, 
                    {"output": "Charming!"})
print("--------------------------------")
print(memory.buffer)
print("--------------------------------")
print(memory.load_memory_variables({}))
