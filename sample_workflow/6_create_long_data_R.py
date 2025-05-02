import json
import pandas as pd
import pathlib
from collections import defaultdict

model = "meta-llama-3.1-70b-instruct"

with open('ChatAI/70B/data/results/logprobs_assoc/'+model+'_res_dict_logprobs_target_expanded.json') as f:
    logprobs_target_dict = json.load(f)


basedata = pd.read_excel('ChatAI/70B/data/database/noun_lists_uni.xlsx', sheet_name="Listen")
np = basedata['NP']
genus = basedata['gender']

genus_dict = dict()
for i, el in enumerate(np):
    genus_dict[el] = genus[i]


np_list = list()
id_list = list()
label_list = list()
np_list = list()
genus_list = list()
logprobs_sum_list = list()
ass_strength_list = list()
total_output_list = list()

n = 0
for key in logprobs_target_dict.keys():
    print(key)
    for l in logprobs_target_dict[key]:
        np_list.append(key)
        genus_list.append(genus_dict[key])
        # for k, v in l.items():
            # np_list.append(key)
        id_list.append(l['id'])
        label_list.append(l['label'])
        logprobs_sum_list.append(l['logprobs_sum'])
        if l['ass_strength_sub'] == "NA":
            ass_strength_list.append("NA")
        else:    
            ass_strength_list.append(l['ass_strength_sub'][0])
        total_output_list.append(l['total_output'])

print(f'np: {len(np_list)} – id: {len(id_list)} – label: {len(label_list)} – logprobs: {len(logprobs_sum_list)} – ass: {len(ass_strength_list)}')

results_long_df = {"NP":np_list, "Genus":genus_list, "ID":id_list, "Label":label_list, "Logprobs_Sum":logprobs_sum_list,  "Association":ass_strength_list, "Total_Output":total_output_list }
results_long_df = pd.DataFrame(results_long_df)
results_long_df.to_csv('ChatAI/70B/data/results/data_for_r/'+model+'_ChatAI_results_long_df.csv', index=False)

