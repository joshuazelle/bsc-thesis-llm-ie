import argparse
import pickle

import pandas as pd
from dotenv import load_dotenv
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

### Setup ###

parser = argparse.ArgumentParser(
    description="Run the model with a custom output filename."
)
parser.add_argument("-output_filename", type=str, required=True, help="Output filename")
parser.add_argument("-model", type=str, required=True, help="exact openai model name")
args = parser.parse_args()


output_filename = args.output_filename
model = args.model

# Load environment variables from .env file
load_dotenv()

llm = ChatOpenAI(model=model, temperature=0)  # take this for longer context length


### Preparation ###


## data prep ##

file = "/home/zelle/development/projects/ascii/reference-data/data_raw_direct_source_drop/joshua/llm_data/df_cc.pickle"

# Load the file
df = pd.read_pickle(file)

# Explode the 'class' column to get one element per row
exploded_class = df["class"].explode()

# Use unique() to get an array of unique values
unique_classes = exploded_class.unique()

# If you need the unique values as a list
categories = list(unique_classes)

# Creating the dictionary with index as key
categories_dict = dict(enumerate(categories))

# initialize predicted class column
df["predicted"] = None
df["predicted"] = df["predicted"].astype(object)


## model prep ##

json_schema = {
    "title": "Company",
    "description": "Identifying information about activities of a company based on its description.",
    "type": "object",
    "properties": {
        "category": {
            "title": "Company's Value Chain Category",
            "description": f"The companies predicted step in Value Chain, it should be part one of the following values: {categories_dict} I want you to just return the numbers of the class!",
            "type": "string",
        },
    },
    "required": ["category"],
}

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an supply chain expert that classifies semiconductor companies based on their activities.",
        ),
        (
            "human",
            "Here is the description of a semiconductor company. Classify the firm based on its activity into one of these steps in the value chain: {input}",
        ),
        (
            "human",
            "Tip: Make sure to answer in the correct format, only return the numbers of your predicted class out of dict before. A firm can be part of up to 4 classes at the same time, seperated with comma. Keep in mind that big firms can make multiple steps while small firms probably specialize on just one step.",
        ),
    ]
)


runnable = create_structured_output_runnable(json_schema, llm, prompt)


# because its more resilient if model outputs the number


def translate_to_categories(input_data):
    # Ensure categories_dict is accessible within the function
    global categories_dict

    # Initialize an empty list to hold the output category names
    output_categories = []

    # Handle string input that represents a dict or comma-separated values
    if isinstance(input_data, str):
        # Try to extract numbers from string assuming it could be a dict or comma-separated
        numbers = [
            int(s)
            for s in input_data.replace("{", "").replace("}", "").split(",")
            if s.isdigit()
        ]
    # Handle list input directly
    elif isinstance(input_data, list):
        numbers = input_data
    else:
        # Return an empty list if the input type is unexpected
        return output_categories

    # Convert numbers to category names, filtering out invalid indices
    for num in numbers:
        if num in categories_dict:
            output_categories.append(categories_dict[num])

    return output_categories


### Run ###

for i in range(len(df)):

    desc = df.iloc[i]["text"]  # get the text

    category_dict = runnable.invoke({"input": desc})
    # Extract the category from the dictionary
    result = category_dict.get("category", "Unknown")

    found_classes = translate_to_categories(result)

    df.at[i, "predicted"] = found_classes

### Save results ###

with open(
    f"/home/zelle/development/projects/ascii/reference-data/data_raw_direct_source_drop/joshua/llm_data/{output_filename}.pickle",
    "wb",
) as f:
    pickle.dump(df, f)
