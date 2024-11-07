from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv,find_dotenv
_=load_dotenv(find_dotenv())

llm=ChatOpenAI(temperature=0.7,api_key=os.getenv("OPENAI_API_KEY"))


prompt_template_1=PromptTemplate.from_template(
    "Translate the following text to American English: {text}"
)
chain_1=LLMChain(llm=llm,prompt=prompt_template_1,output_key="translated_text1")

prompt_template_2=PromptTemplate.from_template(
    "can you summarize the following text in 1 sentence: {translated_text1}"
)
chain_2=LLMChain(llm=llm,prompt=prompt_template_2,output_key="summary")

prompt_template_3=PromptTemplate.from_template(
    "What language is the following text: {summary}"
)
chain_3=LLMChain(llm=llm,prompt=prompt_template_3,output_key="language")

prompt_template_4=PromptTemplate.from_template(
    "Write a follow up message to the following text"
    "Summarize the text in 1 sentence"
    "/n/n Summary: {summary}/n/n Language: {language}"
)
chain_4=LLMChain(llm=llm,prompt=prompt_template_4,output_key="follow_up_message")

overall_chain=SequentialChain(
    chains=[chain_1,chain_2,chain_3,chain_4],
    input_variables=["text"],
    output_variables=["translated_text1","summary","language","follow_up_message"]
)




# Define French text about Python
french_text = """
Python est un langage de programmation polyvalent et puissant.
Il est connu pour sa syntaxe claire et lisible, ce qui le rend facile à apprendre pour les débutants.
Python dispose d'une vaste bibliothèque standard et d'un écosystème riche de packages tiers.
Il est largement utilisé dans des domaines tels que le développement web, l'analyse de données et l'intelligence artificielle.
La communauté Python est active et accueillante, offrant un excellent soutien aux développeurs de tous niveaux.
"""

# Use the overall_chain to process the French text
result = overall_chain.invoke({"text": french_text})

# Print the results
print("Original French text:", french_text)
print("\nTranslated text:", result["translated_text1"])
print("\nSummary:", result["summary"])
print("\nLanguage:", result["language"])
print("\nFollow-up message:", result["follow_up_message"])



