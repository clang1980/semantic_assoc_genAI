import json
import pandas as pd
import pathlib
from itertools import chain
from collections import defaultdict


model = "meta-llama-3.1-70b-instruct"
dir = pathlib.Path("ChatAI/70B/data/results/json")
dir_files = list(dir.rglob("*.json"))
nps = pd.read_excel('ChatAI/70B/data/database/noun_lists_uni.xlsx', sheet_name="Listen")
nps = nps['NP']


def map_subtokens_to_words(subtokens, target_sentence):
    """
    Maps subtokens to their corresponding words in the target sentence.
    
    :param subtokens: List of subtokens (e.g., from a tokenizer).
    :param target_sentence: The full sentence to reconstruct from subtokens.
    :return: A tuple of:
             - positions: List of tuples (start_index, end_index) for each word.
             - grouped_subtokens: List of lists containing subtokens for each word.
    """
    words = target_sentence.split()  # Split sentence into words
    positions = []  # Store start and end indices of subtokens for each word
    grouped_subtokens = []  # Store the subtokens corresponding to each word

    current_word = ""
    current_subtokens = []
    start_index = None
    subtoken_index = 0

    for word in words:
        while subtoken_index < len(subtokens):
            subtoken = subtokens[subtoken_index].strip()
            # Start tracking a new word if needed
            if current_word == "":
                start_index = subtoken_index
            
            # Add subtoken to the current word being formed
            current_word += subtoken
            current_subtokens.append(subtoken)
            
            # Check if the current subtokens form the word
            if current_word == word:
                # Store the positions and subtokens for the completed word
                positions.append((start_index, subtoken_index))
                grouped_subtokens.append(current_subtokens)

                # Reset for the next word
                current_word = ""
                current_subtokens = []
                subtoken_index += 1
                break
            elif current_word in word:  # Continue adding subtokens if we're building the word
                subtoken_index += 1
            else:  # Reset if the current sequence doesn't match
                current_word = ""
                current_subtokens = []
                subtoken_index += 1
    return positions, grouped_subtokens


data_list = []
for file in dir_files:
    file_nr = str(file)
    file_nr = file_nr.replace(".json", '')
    file_nr = file_nr.split("_")[-1]
    with open(file) as f:
        json_data = json.load(f)
        data_list.append(json_data)


res_dict_list = []
n = 0
for np in nps:
    for i, el in enumerate(data_list):
        res_dict = dict()
        res_dict_tmp = dict()
        total_output = el[np]['assoc_multi']['choices'][0]['message']['content']
        logprobs = el[np]['assoc_multi']['choices'][0]['logprobs']['content']
        ass_strength = el[np]['assoc_multi']['choices'][0]['message']['content']
        ass_strength = ass_strength.split("Stärke der Assoziation: ")
        ass_strength = [int(s) for s in ass_strength[-1].split() if s.isdigit()]
        if len(ass_strength) > 1:
            print(f'{i} – {np} – {ass_strength} – ACHTUNG: mehr als ein Wert gefunden!')
        subtokens_tmp = []
        logprobs_tmp = []
        for el in logprobs:
            subtokens_tmp.append(el['token'].strip())
            logprobs_tmp.append(el['logprob'])
            if el['token'] == "ünde":
                break
        subtokens_tmp = [el if el != "Ke" else "ke" for el in subtokens_tmp ]
        target_sequence = "Person weiblichen Geschlechts"
        target_list = [el.strip() for el in target_sequence.split(" ")]
        positions, grouped_subtokens = map_subtokens_to_words(subtokens_tmp, target_sequence)

        if len(positions) < 3:
            target_sequence = "Person männlichen Geschlechts"
            target_list = [el.strip() for el in target_sequence.split(" ")]
            positions, grouped_subtokens = map_subtokens_to_words(subtokens_tmp, target_sequence)

        if len(positions) < 3:
            target_sequence = "keins von beiden"
            target_list = [el.strip() for el in target_sequence.split(" ")]
            positions, grouped_subtokens = map_subtokens_to_words(subtokens_tmp, target_sequence)

        if len(positions) < 3:
            print(f'FEHLER {np} {positions}')
            print(i)
            print("-"*50)
        else:
            pos_list = list(chain.from_iterable(positions))
            pos_list = list(set(pos_list))
            logprobs_list = list()
            label_list = list()
            for el in pos_list:
                logprobs_list.append(logprobs_tmp[el])
                label_list.append(subtokens_tmp[el])
            logprobs_sum = sum(logprobs_list)
            res_dict_tmp['id'] = i
            res_dict_tmp['label'] = target_sequence
            res_dict_tmp['subtokens'] = label_list
            res_dict_tmp['logprobs_sub'] = logprobs_list
            res_dict_tmp['logprobs_sum'] = logprobs_sum
            res_dict_tmp['total_output'] = total_output
            if target_sequence != "keins von beiden":
                if len(ass_strength) == 0:
                    res_dict_tmp['ass_strength_sub'] = "NA" # für Fälle, in denen zwar M oder W gesagt, aber keine Stärke generiert wurde
                elif len(ass_strength) == 2:
                    res_dict_tmp['ass_strength_sub'] = [ass_strength[0]]
                elif len(ass_strength) > 2:
                    res_dict_tmp['ass_strength_sub'] = [ass_strength[-1]] # das ist eine unsaubere Heuristik, um Formulierungen abzufangen wie: auf einer skala von 1 bis 5 bewerte ich mit 4
                elif len(ass_strength) == 1:
                    res_dict_tmp['ass_strength_sub'] = ass_strength
            else:
                res_dict_tmp['ass_strength_sub'] = [0]
        res_dict[np] = res_dict_tmp
        res_dict_list.append(res_dict)        



res_dict_comb = defaultdict(list)
for d in res_dict_list: # you can list as many input dicts as you want here
    for key, value in d.items():
        res_dict_comb[key].append(value)

with open('ChatAI/70B/data/results/logprobs_assoc/'+model+'_res_dict_logprobs_target_expanded.json', 'w', encoding='utf-8') as f:
    json.dump(res_dict_comb, f, ensure_ascii=False, indent=4)


res_dict_logprobs_sum = dict()
for k in res_dict_comb.keys():
    # print(k)
    logprobs_sum_sum = 0
    for el in res_dict_comb[k]:
        # print(el)
        logprobs_sum_sum += el['logprobs_sum']
    # print("_"*50)
    res_dict_logprobs_sum[k] = logprobs_sum_sum

with open('ChatAI/70B/data/results/logprobs_assoc/'+model+'_res_dict_logprobs_sum_np.json', 'w', encoding='utf-8') as f:
    json.dump(res_dict_logprobs_sum, f, ensure_ascii=False, indent=4)

res_dict_assoc_strength_sum = dict()
for k in res_dict_comb.keys():
    assoc_strength_sum = 0
    no_na = 0
    for el in res_dict_comb[k]:
        res_dict_tmp = dict()
        # print(f'{k} - {el['ass_strength_sub']}')
        if el['ass_strength_sub'] != "NA":
            for e in el['ass_strength_sub']:
                assoc_strength_sum += e
        else:
            no_na += 1
    res_dict_tmp['assoc_sum'] = assoc_strength_sum
    res_dict_tmp['assoc_mean'] = assoc_strength_sum/(len(res_dict_comb[k])-no_na)
    res_dict_tmp['no_na'] = no_na
    res_dict_assoc_strength_sum[k] = res_dict_tmp


with open('ChatAI/70B/data/results/logprobs_assoc/'+model+'_res_dict_assoc_strength_sum_np.json', 'w', encoding='utf-8') as f:
    json.dump(res_dict_assoc_strength_sum, f, ensure_ascii=False, indent=4)
