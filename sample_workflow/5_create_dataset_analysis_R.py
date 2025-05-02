import statistics
import json
import pandas as pd


model = "meta-llama-3.1-70b-instruct"

with open('ChatAI/70B/data/results/vgl/'+model+'_itvgl_total.json') as f:
    ratio_dict = json.load(f)

with open('ChatAI/70B/data/results/logprobs_assoc/'+model+'_res_dict_logprobs_sum_np.json') as f:
    logprobs_sum_dict = json.load(f)

with open('ChatAI/70B/data/results/logprobs_assoc/'+model+'_res_dict_assoc_strength_sum_np.json') as f:
    assoc_strength_dict = json.load(f)


# Calculate No Iterations
values = [v for subdict in ratio_dict.values() for v in subdict.values()]
no_iterations = sum(values) / len(ratio_dict)


agreement_dict = dict()
label_ratio_list = list()

maj_vote_dict = dict()
for k in ratio_dict.keys():
    tmp = ratio_dict[k]
    maj_vote_dict[k] = max(tmp, key=tmp.get)

m_list = list()
w_list = list()
n_list = list()
for k in ratio_dict.keys():
    m_list.append(ratio_dict[k]['m'])
    w_list.append(ratio_dict[k]['w'])
    n_list.append(ratio_dict[k]['n'])

    tmp_labels = [v for v in tmp.values()]
    # variance_dict[k] = statistics.variance(tmp_labels)
    ### https://stats.stackexchange.com/questions/164965/raters-agreement-for-each-item
    tmp_list = [ratio_dict[k]['m']] + [ratio_dict[k]['w']] + [ratio_dict[k]['n']]
    agreement_dict[k] = (tmp_list[0]*(tmp_list[0]-1))/(no_iterations*(no_iterations-1)) + (tmp_list[1]*(tmp_list[1]-1))/(no_iterations*(no_iterations-1)) + (tmp_list[2]*(tmp_list[2]-1))/(no_iterations*(no_iterations-1))


logprobs_list = []
agreement_list = []
maj_vote_list = []
assoc_strength_list = []
np_list = []
assoc_strength_list = []
for k in logprobs_sum_dict.keys():
    np_list.append(k)  
    logprobs_list.append(logprobs_sum_dict[k])
    agreement_list.append(agreement_dict[k])
    maj_vote_list.append(maj_vote_dict[k])
    assoc_strength_list.append(assoc_strength_dict[k]['assoc_mean'])


basedata = pd.read_excel('ChatAI/70B/data/database/noun_lists_uni.xlsx', sheet_name="Listen")
np = basedata['NP']
genus = basedata['gender']

genus_dict = dict()
for i, el in enumerate(np):
    genus_dict[el] = genus[i]

genus_list = []
for k in genus_dict.keys():
    genus_list.append(genus_dict[k])

res_comb_df = {'np' : np_list, 'genus' : genus_list, 'maj_vote' : maj_vote_list, 'logprobs' : logprobs_list, 'agreement' : agreement_list, 'assoc_strength_mean' : assoc_strength_list, 'm' : m_list, 'w' : w_list, 'n' : n_list}
res_comb_df = pd.DataFrame(res_comb_df)
res_comb_df.to_csv('ChatAI/70B/data/results/data_for_r/'+model+'_ChatAI_res_logprobs_variance_df.csv', index=False)



