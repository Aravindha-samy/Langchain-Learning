from langchain_community.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv,find_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema,StructuredOutputParser
_=load_dotenv(find_dotenv())





chat=ChatOpenAI(temperature=0,openai_api_key=os.getenv("OPENAI_API_KEY"),model="gpt-3.5-turbo")

#NOTE: Customer Review

customer_review = """\
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""

#NOTE: Prompt Template Without Output Parsers

template_string = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product \
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price, \
and output them as a comma separated Python list.

sentiment: What is the overall sentiment of the review? \
Answer positive, negative, or neutral.

text: {text}
format the output as a JSON with the following keys:
gift,
delivery_days,
price_value,
sentiment
"""

prompt_template=ChatPromptTemplate.from_template(template_string)
# print(prompt_template)
# print("****************************************************************")
# print(prompt_template.messages)
# print("****************************************************************")
# print(prompt_template.messages[0].prompt)
# print("****************************************************************")
# print(prompt_template.messages[0].prompt.input_variables)
# print("****************************************************************")
messages=prompt_template.format_messages(text=customer_review)

response=chat.invoke(messages)
print(response)
print("****************************************************************")
print(response.content)

#NOTE: Prompt Template With Output Parsers 
print(ok)

template_string = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product \
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price, \
and output them as a comma separated Python list.

sentiment: What is the overall sentiment of the review? \
Answer positive, negative, or neutral.

text: {text}

{format_instructions}
"""


#IMPORTANT: Output Parsers
sentiment = ResponseSchema(name="sentiment", description="What is the overall sentiment of the review? Answer positive, negative, or neutral.")
gift_format=ResponseSchema(name="gift",description="Was the item purchased as a gift for someone else? Answer True if yes, False if not or unknown.")
delivery_days=ResponseSchema(name="delivery_days",description="How many days did it take for the product to arrive? If this information is not found, output -1.")
price_value=ResponseSchema(name="price_value",description="Extract any sentences about the value or price, and output them as a comma separated Python list.")

response_schemas=[gift_format,delivery_days,price_value,sentiment]
output_parser=StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions=output_parser.get_format_instructions() #get the format instructions as a Json

prompt_template=ChatPromptTemplate.from_template(template_string)

messages=prompt_template.format_messages(text=customer_review,format_instructions=format_instructions)

response=chat.invoke(messages)
print(response.content)

parsed=output_parser.parse(response.content)
print(parsed)
print(parsed.get("gift"))
print(parsed.get("delivery_days"))
print(parsed.get("price_value"))
print(parsed.get("sentiment"))


