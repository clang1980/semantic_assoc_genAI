import json
import pandas as pd
import pathlib
import re

model = "meta-llama-3.1-70b-instruct"

dir = pathlib.Path("ChatAI/70B/data/results/json")

dir_files = list(dir.rglob("*.json"))

### read candidate NP
words = pd.read_excel('ChatAI/70B/data/database/noun_lists_uni.xlsx', sheet_name="Listen")
words = words['NP']


for json_file in dir_files:
    filename = str(json_file)
    print(filename)
    filename_csv = filename.replace("json", "csv")
    if model in filename_csv:
        # print(filename_csv)
        ### read openai res data
        with open(filename) as f:
            results_assoc = json.load(f)
        # print(results_assoc)
        np_list = []
        assoc_result = []

        for word in words:
            np_list.append(word)
            assoc_result.append(results_assoc[word]['assoc_multi']['choices'][0]['message']['content'])


        label = []
        reasons = []
        score = []
        score_reason = []
        for el in assoc_result:
            el = el.replace("*", "")
            el = el.replace("\n", " ")
            parts1 = re.split('Gründe', el)
            label_tmp = re.sub(r"Kurzantwort","", parts1[0])
            label_tmp = re.sub(r"##","", label_tmp).strip()
            if label_tmp == "Keins von beiden":
                label_tmp = "keins von beiden"
            parts2 = re.split('Stärke der Assoziation', parts1[1:][0])
            reason_tmp = parts2[0].strip()
            if label_tmp == "Person männlichen Geschlechts" or label_tmp == "Person weiblichen Geschlechts":
                try:
                    score_reason_tmp = parts2[1].strip()
                    for char in score_reason_tmp:
                        if char.isnumeric():
                            score_tmp = char
                except:
                    score_tmp = "NA"
                print('Label: ' + label_tmp)
                print("-"*50)
            else:
                score_reason_tmp = ""
                score_tmp = ""
            label.append(label_tmp)
            reasons.append(reason_tmp)
            score.append(score_tmp)
            score_reason.append(score_reason_tmp)

        results_assoc_text_df = {"NP":np_list, "label":label, "reasons":reasons, "score":score, "score_reason":score_reason, "total_output":assoc_result}
        results_assoc_text_df = pd.DataFrame(results_assoc_text_df)
        results_assoc_text_df.to_csv(filename_csv, index=False)
