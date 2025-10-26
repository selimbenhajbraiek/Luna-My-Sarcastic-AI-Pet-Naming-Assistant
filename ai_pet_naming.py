"""
Author: Selim Ben Haj Braiek
Project: AI Pet Naming Assistant
Description: Simple Sarcastic AI Pet Naming Assistant
  
"""

# Load environment variables (for OpenAI API key)
%load_ext dotenv
%dotenv

# Import necessary classes from LangChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)

# Initialize the chat model
chat_model = ChatOpenAI(
    model='gpt-4o',
    model_kwargs={'seed': 42},
    temperature=0,
    max_tokens=100
)

#  Define prompt templates
SYSTEM_TEMPLATE = '{bot_description}'
HUMAN_TEMPLATE = '''I've recently adopted a {animal}.
Could you suggest some {animal} names?'''

# Create message templates
system_prompt = SystemMessagePromptTemplate.from_template(template=SYSTEM_TEMPLATE)
human_prompt = HumanMessagePromptTemplate.from_template(template=HUMAN_TEMPLATE)

#  Combine into a chat prompt template
chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

#  Provide dynamic input values
prompt_inputs = {
    'bot_description': 'You are Luna, a witty and slightly sarcastic virtual pet-naming assistant.',
    'animal': 'cat'
}

#  Generate the structured chat input
chat_input = chat_prompt.invoke(prompt_inputs)

#  Get model response
response = chat_model.invoke(chat_input)

#  Display output
print(response.content)
