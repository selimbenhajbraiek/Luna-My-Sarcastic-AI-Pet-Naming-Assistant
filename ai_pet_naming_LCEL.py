"""
Author: Selim Ben Haj Braiek
Project: AI Pet Naming Assistant
Description: Simple Sarcastic AI Pet Naming Assistant using LCEL with batching and streaming capabilities.
  
"""

# Load environment variables for OpenAI API key
%load_ext dotenv
%dotenv

# Import necessary classes
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser

# 1️ Initialize the chat model
chat_model = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=100,
    model_kwargs={"seed": 42}
)


# 2️ Define the chat prompt template
chat_template = ChatPromptTemplate.from_messages([
    ("system", "{bot_description}"),
    ("human", "I've recently adopted a {animal}. Could you suggest some {animal} names?")
])

# 3️ Define the output parser

list_output_parser = CommaSeparatedListOutputParser()

# 4️ Create the processing chain
chain = chat_template | chat_model | list_output_parser

# 5️ Batch processing and streaming examples
batch_inputs = [
    {
        "bot_description": "You are Luna, a witty and slightly sarcastic virtual pet-naming assistant.",
        "animal": "cat"
    },
    {
        "bot_description": "You are Luna, a witty and slightly sarcastic virtual pet-naming assistant.",
        "animal": "dog"
    },
    {
        "bot_description": "You are Luna, a witty and slightly sarcastic virtual pet-naming assistant.",
        "animal": "parrot"
    }
]

# 6️ Execute batch processing
batch_results = chain.batch(batch_inputs)

for animal_input, names in zip(batch_inputs, batch_results):
    print(f"{animal_input['animal'].title()} names: {names}")

# 7️ Execute streaming processing
print("\n=== STREAMING RESULTS ===")
stream_input = {
    "bot_description": "You are Luna, a witty and slightly sarcastic virtual pet-naming assistant.",
    "animal": "hamster"
}
#  Initialize an empty string to collect streamed results
streamed_result = ""
for chunk in chain.stream(stream_input):
    print(chunk, end="", flush=True)
    streamed_result += chunk

