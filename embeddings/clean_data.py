import os
import json
import zipfile

# unzip downloaded file with the dataset
with zipfile.ZipFile(os.path.join(os.path.dirname(__file__), "en_medical_dialog.json"), 'r') as zip_ref:
    zip_ref.extractall()

# delete remaining zip and feather files, leaving only the resulting json
os.remove(os.path.join(os.path.dirname(__file__), "archive.zip"))
os.remove(os.path.join(os.path.dirname(__file__), "medical_dialogues_cleaned.feather"))

# import json as dict
with open(os.path.join(os.path.dirname(__file__), "en_medical_dialog.json")) as json_file:
    data = json.load(json_file)

for row in data:
    row['tokens'] = len(row['Doctor'].split())
    if row['tokens'] > 250:
        data.remove(row)
    elif row['tokens'] < 25:
        data.remove(row)
    if row["id"] % 1000 == 0:
        print(row["id"], "rows processed")

# export dict to json
with open('embeddings/medical_dialogues_cleaned.json', 'w') as fp:
    json.dump(data, fp)

