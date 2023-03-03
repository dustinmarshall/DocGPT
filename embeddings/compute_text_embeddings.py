import openai
import pandas as pd
import time
import json
import os

# set up the API key
openai.api_key = "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"

# import pandas dataframe from jsonl file
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "embeddings/medical_dialogues_cleaned.csv"))

def get_embedding(text: str, model: str="text-embedding-ada-002") -> list[float]:
    retry_count = 0
    while retry_count < 3:
        try:
            result = openai.Embedding.create(
              model=model,
              input=text
            )
            return result["data"][0]["embedding"]
        except Exception as e:
            retry_count += 1
            if retry_count == 3:
                print(e)
            print("Caught exception, retrying in 5 seconds...")
            time.sleep(5)

def compute_doc_embeddings(df: pd.DataFrame) -> dict[str, list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    embeddings = {}
    count = 0
    for idx, r in df.iterrows():
        embeddings[idx] = get_embedding(r.description_question)
        count += 1
        if count % 500 == 0:
            print(count, "rows processed")
    return embeddings

# compute document embeddings
document_embeddings = compute_doc_embeddings(df)

# export dict to json
with open('text_embeddings.json', 'w') as fp:
    json.dump(document_embeddings, fp)