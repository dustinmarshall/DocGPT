import os
import json

# import json as dict (download from https://www.kaggle.com/datasets/dsxavier/diagnoise-me)
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

