from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
tools = load_tools(["llm-math","wikipedia"], llm=llm)

agent= initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)
print(agent("What is the 25% of 300?"))

question = "What is the population of India and who will be the Chief Minister of TamilNadu in 2026"
result = agent(question) 

# Python Agent

agent = create_python_agent(
    llm,
    tool=PythonREPLTool(),
    verbose=True
)

customer_list = [["Harrison", "Chase"], 
                 ["Lang", "Chain"],
                 ["Dolly", "Too"],
                 ["Elle", "Elem"], 
                 ["Geoff","Fusion"], 
                 ["Trance","Former"],
                 ["Jen","Ayai"]
                ]
import langchain
langchain.debug=True

agent.run(f"""Sort these customers by \
last name and then first name \
and print the output: {customer_list}""") 

# Defining the Own tool

from langchain.agents import tool
from datetime import date

@tool
def time(text: str) -> str:
    """
    Returns todays date, use this for any \
    questions related to knowing todays date. \
    The input should always be an empty string, \
    and this function will always return todays \
    date - any date mathematics should occur \
    outside this function."""
    return str(date.today())

agent= initialize_agent(
    tools + [time], 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)

try:
    result = agent("whats the date today?") 
except: 
    print("exception on external access")

print(result)