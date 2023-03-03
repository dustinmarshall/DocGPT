from flask import Flask, render_template, request
import openai
import pandas as pd
import json
import time
import numpy as np
import tiktoken
import os

app = Flask(__name__)

# set up the API key
openai.api_key = "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"

# import pandas dataframe from jsonl file
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "embeddings/medical_dialogues_cleaned.csv"))

# import json file as dict
with open(os.path.join(os.path.dirname(__file__), "embeddings/text_embeddings.json")) as json_file:
    document_embeddings = json.load(json_file)  

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

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: dict[(str), np.array]) -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[int(section_index)]
        
        chosen_sections_len += document_section.tokens + len(tiktoken.get_encoding("gpt2").encode("\n* "))
        if chosen_sections_len > 500:
            break
            
        chosen_sections.append("\n* " + document_section.answer)
        chosen_sections_indexes.append(str(section_index))
    
    header = """Answer the question as truthfully as possible using the provided context, and if the answer 
    is not contained within the text below, say "I'm sorry, that isn't within my expertise."\n\nContext:\n"""
    
    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"

COMPLETIONS_API_PARAMS = {
    "temperature": 0.0,
    "max_tokens": 100,
    "model": "text-davinci-003",
}

def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[str, np.array],
    show_prompt: bool = False
) -> str:
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )
    
    if show_prompt:
        print(prompt)

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )

    return response["choices"][0]["text"].strip(" \n")

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the text from the input form
        text = request.form['text']
        # Call the reverse_text function on the input text
        answer = answer_query_with_context(text, df, document_embeddings)
        # Return the reversed text to the user
        return render_template('index.html', answer=answer, text=text)
    else:
        # If the user hasn't submitted any text yet, just show the empty form
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)