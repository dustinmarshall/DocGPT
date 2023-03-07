# DocGPT : Medical Q&A Chatbot
This repository contains the source code for a chatbot application built using OpenAI's Completions and Embedding APIs, as well as a dataset of 257k doctor-patient dialogs.

# Description
DocGPT is currently only a prototype. The truth value of it's responses still needs to be thoroughly validated. The project was concieved of as a potential intervention to be piloted and evaluated for the final project of a graduate course on Big Data and Development taught at UChicago in winter of 2023.

# Installation
To replicate this chatbot application, follow these steps (requires a Kaggle account for access to the dataset, a Pinecone account for storing the embeddings, and a Heroku account account for hosting the application):

1. Clone this repository to your local machine
2. Open a terminal and navigate to where the repository is saved
3. Run the following code from the command line to securely save private variables associated with your Kaggle, OpenAI, and Pinecone accounts to your local environment:
    `export KAGGLE_KEY=YOUR-KEY-HERE`
    `export OPENAI_API_KEY=YOUR-KEY-HERE`
    `export PINECONE_API_KEY=YOUR-KEY-HERE`
    `export PINECONE_ENVIRONMENT=YOUR-ENVIRONMENT-HERE`
4. To download the dataset from Kaggle, run the following code from the command line:
    `kaggle datasets download -d dsxavier/diagnoise-me`
5. To clean the doctor-patient dialog data, run the following code from the command line:
    `python3 /embeddings/clean_data.py`
6. To compute the embeddings and store them in your Pinecone Index, run the following code from the command line:
    `python3 /embeddings/compute_embeddings.py`
7. To create the app on Heroku and link it to your existing GitHub repo, run the following code:
    `python3 /application/create_app.py`

# Contact
If you have any questions or concerns, please feel free to reach out to Dustin Marshall (dustinmarshall@uchicago.edu).
