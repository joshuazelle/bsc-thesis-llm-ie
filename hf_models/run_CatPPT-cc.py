import pickle

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

### Setup ###

output_filename = "CatPPT_cc_0"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained("rishiraj/CatPPT")

tokenizer = AutoTokenizer.from_pretrained("rishiraj/CatPPT")

model.to(device)

### Functions ###


def call_llm(text):
    messages = [{"role": "user", "content": text}]
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

    model_inputs = encodeds.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)

    decoded = tokenizer.batch_decode(generated_ids)

    # Split the decoded text on the [/INST] token and return the part after it
    response = decoded[0].split("[/INST]")[-1].strip()

    # Further trim the "</s>" token from the end of the response if it exists
    if response.endswith("</s>"):
        response = response[:-4].strip()

    return response


# because sometimes the model returns more than just the class
def find_categories_in_text(categories, text):
    found_categories = []
    for category in categories:
        if category in text:
            found_categories.append(category)
    return found_categories


### Preparation ###

file = "/home/zelle/development/projects/ascii/reference-data/data_raw_direct_source_drop/joshua/llm_data/df_cc.pickle"

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

    description = df.iloc[i]["extr_text"]  # get the orbis text

    prompt = f"""Here is a text from a semiconductor company website. Can you infer from this text what the company does? Classify it into these {len(categories)} classes which should be self-explanatory: {categories}.
        A firm can be part of 1 or many classes. For instance big firms could do multiple things while smaller firms tend to specialize in one class.
        Return your predicted class(es) as a python list with strings, e.g.: ['Design'] or ['tool_ressource','Fabrication']. Return only this list, nothing else.
        Here the website text:
        {description}


        REMINDER: only return your predicted class out of these {categories}, DO NOT use any other categories, and DO NOT return anything except the predicted list.
        """

    result = call_llm(prompt)

    found_classes = find_categories_in_text(categories, result)

    df.at[i, "predicted"] = found_classes

    if i % 10 == 0:
        print(i)

### Save results ###

with open(
    f"/home/zelle/development/projects/ascii/reference-data/data_raw_direct_source_drop/joshua/llm_data/{output_filename}.pickle",
    "wb",
) as f:
    pickle.dump(df, f)
