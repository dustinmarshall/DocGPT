from flask import Flask, render_template, request
import openai
import time
import pinecone
import os

app = Flask(__name__)

# set up the API key
openai.api_key = os.environ.get('OPENAI_API_KEY') 

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

def construct_prompt(query: str) -> str:
    """
    Fetch relevant 
    """
    pinecone.init(api_key="e5a609b1-133e-4c44-8629-51c9bb3a6104", environment="us-east1-gcp")
    index = pinecone.Index("medical-dialog-embeddings")
    query_embedding = get_embedding(query,model="text-embedding-ada-002")
    most_relevant_answers = index.query([query_embedding], top_k=5, include_metadata=True)

    answers_to_insert = []
    for i in most_relevant_answers["matches"]:
        answers_to_insert.append("\n- " + i["metadata"]["answer_raw"])
    
    header = """\n\nAnswer the question above as truthfully as possible using the context below. 
    If the answer is not contained within the context below, say "I'm sorry, that isn't within 
    my expertise."\n\n"""
    
    return "Q: " + query + header + "".join(answers_to_insert) + "\n\nA: "

COMPLETIONS_API_PARAMS = {
    "temperature": 0.0,
    "max_tokens": 100,
    "model": "text-davinci-003",
}

def answer_query_with_context(query: str, show_prompt: bool = False) -> str:
    
    prompt = construct_prompt(query)
    
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
        answer = answer_query_with_context(text)
        # Return the reversed text to the user
        return render_template('index.html', answer=answer, text=text)
    else:
        # If the user hasn't submitted any text yet, just show the empty form
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)