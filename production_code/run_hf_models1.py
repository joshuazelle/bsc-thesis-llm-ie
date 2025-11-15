import argparse
import logging
import os
import pickle

import pandas as pd
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer


### Load environment variables from the .env file ###
def load_environment_variables():
    load_dotenv()


### Setup logging ###
def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


### Command line arguments ###
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run the model with custom input file and output filename."
    )
    parser.add_argument(
        "-input_filepath",
        type=str,
        required=True,
        help="Input file path, needs to be a pickle dataframe with columns 'text' for prediction and 'class' for the true class of the firm.",
    )
    parser.add_argument(
        "-output_filename",
        type=str,
        required=True,
        help="Output filename as it will be saved to OUTPUT_DIR.",
    )
    parser.add_argument(
        "-model",
        type=str,
        required=True,
        help="Exact Hugging Face model name, e.g., 'mistralai/Mistral-7B-Instruct-v0.2'.",
    )
    return parser.parse_args()


### Model and tokenizer setup ###
def initialize_model_and_tokenizer(model_name):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    return model, tokenizer, device


### Call the model ###
def call_model(model, tokenizer, device, description, categories):
    """
    Generate a response from the model based on the provided description and categories.

    This function constructs a prompt using the given description and categories, sends it to the model,
    and processes the model's output to extract and return the relevant response.

    Args:
    - model (PreTrainedModel): The loaded model object from Hugging Face's transformers.
    - tokenizer (PreTrainedTokenizer): The tokenizer corresponding to the model, used for encoding the input text.
    - device (str): The device type (e.g., 'cuda:0', 'cpu') indicating where the model is loaded.
    - description (str): The text description from a company to be analyzed by the model.
    - categories (list): The list of categories which should be predicted.

    Returns:
    - list: A list of strings, each representing a category identified from the model's response to the prompt.
    """

    prompt = f"""Here is a text from a semiconductor company website. Can you infer from this text what the company does? Classify it into these {len(categories)} classes which should be self-explanatory: {categories}.
    A firm can be part of 1 or many classes. For instance, big firms could engage in multiple activities while smaller firms tend to specialize in one class.
    Return your predicted class(es) as a python list with strings, e.g., ['Design'] or ['tool_resource', 'Fabrication']. Return only this list, nothing else.
    Here the website text:
    {description}

    REMINDER: only return your predicted class out of these {categories}, DO NOT use any other categories, and DO NOT return anything except the predicted list.
    """
    messages = [{"role": "user", "content": prompt}]
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to(device)
    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    response = decoded[0].split("[/INST]")[-1].strip()
    if response.endswith("</s>"):
        response = response[:-4].strip()
    return response


### Find categories within the returned text ###
def find_categories_in_text(categories, text):
    return [category for category in categories if category in text]


### Main execution function ###
def main():
    setup_logging()
    args = parse_arguments()
    load_environment_variables()

    # Check if OUTPUT_DIR is set in the environment and valid before proceeding
    output_dir = os.getenv("OUTPUT_DIR")
    if not output_dir:
        logging.error(
            "OUTPUT_DIR environment variable is not set. Falling back to the current working directory."
        )
        output_dir = os.getcwd()

    output_path = os.path.join(output_dir, f"{args.output_filename}.pickle")

    logging.info("Loading data...")
    df = pd.read_pickle(args.input_filepath)

    logging.info(f"Preparing model and tokenizer for {args.model}...")
    model, tokenizer, device = initialize_model_and_tokenizer(args.model)

    exploded_class = df["class"].explode()
    categories = list(exploded_class.unique())
    df["predicted"] = None

    logging.info("Classifying descriptions...")
    for i, row in df.iterrows():
        if i % 10 == 0:
            logging.info(f"{i} companies processed...")
        result = call_model(model, tokenizer, device, row["text"], categories)
        found_classes = find_categories_in_text(categories, result)
        df.at[i, "predicted"] = found_classes

    logging.info(f"Saving results to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(df, f)

    logging.info("Processing completed successfully.")


if __name__ == "__main__":
    main()
