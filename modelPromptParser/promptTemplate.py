from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv,find_dotenv
from langchain.prompts import ChatPromptTemplate
import os
_=load_dotenv(find_dotenv())

#Set OpenAI API Key
llm = ChatOpenAI(temperature=0,openai_api_key=os.getenv("OPENAI_API_KEY"),model="gpt-3.5-turbo")


#Common Template for Prompt Engineering

template_string= """
Translate the text in language {language} that is delimited by triple backticks
into a style that is {style}
text: ```{text}```
"""
#Create Prompt Template
prompt_template=ChatPromptTemplate.from_template(template_string);

print(prompt_template.messages[0].prompt.input_variables)

#Create Variables
common_language=str(input("Enter the language: "))

customer_style="""In a calm way and respectful tone"""
customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse, \
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

service_reply = """Hey there customer, \
the warranty does not cover \
cleaning expenses for your kitchen \
because it's your fault that \
you misused your blender \
by forgetting to put the lid on before \
starting the blender. \
Tough luck! See ya!
"""

service_style_pirate = """\
a polite tone \
that speaks in English Pirate\
"""
service_messages=prompt_template.format_messages(
    language=common_language,
    style=service_style_pirate,
    text=service_reply
)


customer_messages=prompt_template.format_messages(
    language=common_language,
    style=customer_style,
    text=customer_email
)
customer_response=llm.invoke(customer_messages)
service_response=llm.invoke(service_messages)

print(customer_response.content)
print(service_response.content)

