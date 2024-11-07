import openai
from openai import OpenAI
import datetime
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#Note: This is normal AI call
llm_model = "gpt-3.5-turbo"

# NOTE: get completion
def get_completion(prompt, model=llm_model, temperature=0):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    print (response)
    print (response.choices)
    return response.choices[0].message.content


language=str(input("Enter Your Language"))

style=f""" in a calm and respectful tone"""

customer_email="""Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse,\
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!"""

prompt=f"""Translate the text to {language} \
    that is delimited by triple backticks \
    into that style: {style}
    text:```{customer_email}```
"""

response=get_completion(prompt)

print(response)

