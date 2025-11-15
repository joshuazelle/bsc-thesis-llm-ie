import argparse
import logging
import os
import pickle

import pandas as pd
from dotenv import load_dotenv
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# Setup logging configuration
def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


# Load environment variables from the .env file
def load_environment_variables():
    load_dotenv()


# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run the model with custom inputfile (full path) and output filename (only name)."
    )
    parser.add_argument(
        "-input_filepath",
        type=str,
        required=True,
        help="Input file path, needs to be pickle dataframe with columns 'text' that will be used for prediction and 'class' that is the true class of the firm.",
    )
    parser.add_argument(
        "-output_filename",
        type=str,
        required=True,
        help="Output filename as it will be saved to OUTPUT_DIR",
    )
    parser.add_argument(
        "-model",
        type=str,
        required=True,
        help="Exact OpenAI model name, for instance 'gpt-3.5-turbo' ",
    )
    return parser.parse_args()


# Load data from a given file path
def load_data(file_path):
    return pd.read_pickle(file_path)


# Prepare categories from the dataframe
def prepare_categories(df):
    exploded_class = df["class"].explode()
    unique_classes = exploded_class.unique()
    return dict(enumerate(unique_classes))


# Create a model instance for structured output generation
def create_model(model_name, categories_dict):
    """
    Create a model instance using LangChain and OpenAI with a specific structure defined for JSON schema.

    This function initializes a structured output runnable with a specific prompt template and JSON schema,
    tailored for classifying company activities into specific categories. The categories_dict is used to
    dynamically insert the possible categories into the JSON schema description, ensuring the model's output
    is interpretable and aligned with the expected classification categories.

    Args:
    - model_name (str): The name of the model to be used from OpenAI.
    - categories_dict (dict): A dictionary mapping category indices to their descriptive names.

    Returns:
    - A runnable instance configured to generate structured outputs based on the given model and schema.
    """
    json_schema = {
        "title": "Company",
        "description": "Identifying information about activities of a company based on its description.",
        "type": "object",
        "properties": {
            "category": {
                "title": "Company's Value Chain Category",
                "description": f"The company's predicted step in the Value Chain should be one of the following values: {categories_dict}. Please return only the class numbers.",
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
                "Tip: Make sure to answer in the correct format, only returning the numbers of your predicted class. Remember, a firm can be classified into multiple categories.",
            ),
        ]
    )

    llm = ChatOpenAI(model=model_name, temperature=0)
    return create_structured_output_runnable(json_schema, llm, prompt)


# Translate numerical categories to their descriptive names
def translate_to_categories(input_data, categories_dict):
    numbers = [
        int(s)
        for s in str(input_data).replace("{", "").replace("}", "").split(",")
        if s.isdigit()
    ]
    return [categories_dict.get(num) for num in numbers if num in categories_dict]


# The main execution function
def main():
    setup_logging()
    load_environment_variables()
    args = parse_arguments()

    # Check if OUTPUT_DIR is set in the environment and valid before proceeding
    output_dir = os.getenv("OUTPUT_DIR")
    if not output_dir:
        logging.error(
            "OUTPUT_DIR environment variable is not set. Falling back to the current working directory."
        )
        output_dir = os.getcwd()

    output_path = os.path.join(output_dir, f"{args.output_filename}.pickle")

    logging.info("Loading data...")
    df = load_data(args.input_filepath)
    categories_dict = prepare_categories(df)

    logging.info("Setting up model...")
    model = create_model(args.model, categories_dict)

    logging.info("Start analysis...")
    for i, row in df.iterrows():
        if i % 10 == 0:
            logging.info(f"{i} companies processed...")
        result = model.invoke({"input": row["text"]}).get("category", "Unknown")
        df.at[i, "predicted"] = translate_to_categories(result, categories_dict)

    with open(output_path, "wb") as f:
        pickle.dump(df, f)

    logging.info("Processing completed successfully.")


if __name__ == "__main__":
    main()
