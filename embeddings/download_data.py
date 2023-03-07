import zipfile
import os

zip_file_path = "/Users/dustin/Documents/GitHub/DocGPT/embeddings/archive.zip"  # Replace with the path to your ZIP file

with zipfile.ZipFile(os.path.join(os.path.dirname(__file__), "archive.zip"), 'r') as zip_ref:
    zip_ref.extractall(os.path.dirname(__file__))

print(os.path.dirname(__file__))
