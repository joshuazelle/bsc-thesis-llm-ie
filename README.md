# Supply Chain Information Extraction using LLMs

## About This Project

This repository contains code from my BSc thesis "Extracting supply chain information from text data using LLMs", completed in March 2024 at Vienna University of Economics and Business (WU Wien), in collaboration with the [Complexity Science Hub Vienna](https://csh.ac.at/) and the [Supply Chain Intelligence Institute Austria](https://ascii.ac.at/).

**Note on Reproducibility:** The data used in this project is proprietary and not publicly available. This repository serves as a code portfolio demonstrating the methodology and implementation. The full thesis PDF is available [here](thesis.pdf).

## Overview

The most important files are: the [LLM implementations](production_code/), the [evaluation results](hf_models/eval.ipynb), or the [Common Crawl pipeline](commoncrawl/)

Here is a brief overview of what can be found in each folder:

- [`commoncrawl`](commoncrawl/):
    Pipeline for web data extraction and processing:
    1. Get URLs from the [Georgetown seed companies](https://github.com/georgetown-cset/eto-chip-explorer/blob/main/data/providers.csv) that were crawled
    2. Filter all URLs available in the crawl and compute basic statistics
    3. Get HTML content from commoncrawl for those filtered URLs
    4. Extract text from HTML 
    5. AI translation experiments (ultimately not used as models were multilingual)
    6. Prepare data for LLMs

- [`hf_models`](hf_models/):
    Implementations for running Hugging Face models locally on GPU. The most important file is [eval.ipynb](hf_models/eval.ipynb) where the evaluation function lives.

- [`openai_runs`](openai_runs/):
    Implementations for OpenAI API using langchain

- [`other`](other/):
    Experimental code and alternative approaches not used in final implementation

- [`production_code`](production_code/):
    Python scripts for running classification at scale via command line interface


## Requirements

This code was developed and run in ASCII's computing environment. To run locally, you would need:

- Python 3.8+
- GPU with CUDA support (for local model inference)
- Transformers library (Hugging Face)
- OpenAI API key (for GPT models)
- Standard data science libraries (pandas, numpy, etc.)

See individual scripts for specific dependencies.

## Abstract

Large Language Models (LLMs) have garnered significant attention in research, particularly their application to core natural language processing (NLP) tasks like information extraction (IE), which has seen a notable uptick in interest recently. Moreover, the deployment of LLMs across specialized fields, including medicine and industry, is expanding. A promising opportunity lies in supply chain analysis, an area where traditional methods often fall short in unraveling the complexities of modern, often non-transparent supply networks. Applying NLP advancements to these challenges in supply chain analysis, this thesis investigates the question: Can LLMs accurately extract supply chain information from text data? It delves into this through two sub-questions: (1) how do different LLMs compare on this task and (2) how is the performance given different input texts?

To address these questions, we employ a ground truth classification framework within the semiconductor industry as a case study. Additionally, two custom datasets were curated as input for the LLMs: one derived from company descriptions sourced from the business intelligence database Orbis, and the other extracted from text found on company websites, utilizing the web-scale data archive Common Crawl. The LLMs were tasked with classifying semiconductor companies into stylized segments of the supply chain, thereby transforming the task into a hybrid problem involving IE and multi-label classification.

An interesting implication of this approach is that, if successfully tested against a ground truth, it could then be employed in novel scenarios, potentially uncovering previously unknown insights into supply chains. The results demonstrate that while LLMs show promise in this domain, their performance is significantly influenced by the source text quality. The study highlights the importance of general reasoning abilities in LLMs even in task-specific models, at least for more complex scenarios. It also identifies key challenges in the usage of LLMs combined with targeted web data extraction, setting a foundation for future advancements of such LLM applications in general and for AI-driven supply chain analysis in particular.