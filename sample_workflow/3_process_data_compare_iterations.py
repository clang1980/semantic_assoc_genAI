import pandas as pd
import pathlib
import json

model = "meta-llama-3.1-70b-instruct"

dir = pathlib.Path("ChatAI/70B/data/results/csv")
dir_files = list(dir.rglob("*.csv"))
words = pd.read_excel('ChatAI/70B/data/database/noun_lists_uni.xlsx', sheet_name="Listen")
words = words['NP']


### alle 40 Trials
res_dict_total = dict()
for np in words:
    res_dict_total[np] = {'m':0, 'w':0, 'n':0}

for i, file in enumerate(dir_files):
    print(i)
    # print(file)
    filename = str(file)
    res_df = pd.read_csv(filename)
    nps = res_df['NP']
    label = res_df['label']
    it = [i]*len(label)
    for i, np in enumerate(nps):
        label_tmp = label[i]
        label_tmp = label_tmp.replace("##", '').strip()
        print(label_tmp)
        if label_tmp == "Person männlichen Geschlechts":
            res_dict_total[np]['m'] += 1
        elif label_tmp == "Person weiblichen Geschlechts":
            res_dict_total[np]['w'] += 1
        elif label_tmp == "keins von beiden":
            res_dict_total[np]['n'] += 1
        else:
            print(f'Problem: _{label_tmp}_')
    print("-"*50)


with open('ChatAI/70B/data/results/vgl/'+model+'_itvgl_total.json', 'w', encoding='utf-8') as f:
    json.dump(res_dict_total, f, ensure_ascii=False, indent=4)    

