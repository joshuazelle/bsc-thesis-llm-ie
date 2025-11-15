import argparse
import pickle

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAI

### Setup ###

parser = argparse.ArgumentParser(
    description="Run the model with a custom output filename."
)
parser.add_argument("-output_filename", type=str, required=True, help="Output filename")
args = parser.parse_args()


output_filename = args.output_filename

model = "gpt-3.5-turbo-instruct"

# Load environment variables from .env file
load_dotenv()

llm = OpenAI(model=model, temperature=0)  # take this for longer context length


### Functions ###


def call_model(text):

    return llm.predict(text)


# because sometimes the model returns more than just the class
def find_categories_in_text(categories, text):
    found_categories = []
    for category in categories:
        if category in text:
            found_categories.append(category)
    return found_categories


### Preparation ###

file = "/home/zelle/development/projects/ascii/reference-data/data_raw_direct_source_drop/joshua/llm_data/gt_orb.pickle"

# Load the file
df = pd.read_pickle(file)

# Explode the 'class' column to get one element per row
exploded_class = df["class"].explode()

# Use unique() to get an array of unique values
unique_classes = exploded_class.unique()

# If you need the unique values as a list
categories = list(unique_classes)

# initialize predicted class column
df["predicted"] = None
df["predicted"] = df["predicted"].astype(object)


### Execute ###

for i in range(len(df)):

    description = df.iloc[i]["orbis_description"]  # get the orbis text

    prompt = f"""Here is a text from a semiconductor company website. Can you infer from this text what the company does? Classify it into these {len(categories)} classes which should be self-explanatory: {categories}.
        A firm can be part of 1 or many classes. For instance big firms could do multiple things while smaller firms tend to specialize in one class.
        Return your predicted class(es) as a python list with strings, e.g.: ['Design'] or ['tool_ressource','Fabrication']. Return only this list, nothing else.
        Here the website text:
        {description}


        REMINDER: only return your predicted class out of these {categories}, DO NOT use any other categories, and DO NOT return anything except the predicted list.
        """

    result = call_model(prompt)

    found_classes = find_categories_in_text(categories, result)

    df.at[i, "predicted"] = found_classes

### Save results ###

with open(
    f"/home/zelle/development/projects/ascii/reference-data/data_raw_direct_source_drop/joshua/llm_data/{output_filename}.pickle",
    "wb",
) as f:
    pickle.dump(df, f)
