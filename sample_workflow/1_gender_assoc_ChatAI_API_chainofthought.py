
from openai import OpenAI
import os
import json
from dotenv import load_dotenv
import pandas as pd
import time

start_time = time.time()


### read candidate NP
words = pd.read_excel('ChatAI/70B/data/database/noun_lists_uni.xlsx', sheet_name="Listen")
words = words['NP']

# API configuration
api_key = '<apikey>' # Replace with your API key
base_url = "https://chat-ai.academiccloud.de/v1"
model = "meta-llama-3.1-70b-instruct" # Choose any available model
  
from openai import OpenAI

it = range(1,34) # generate 33 labels

for i in it:
    client = OpenAI(
    api_key = api_key,
    base_url = base_url
)

    results_assoc_multi = dict()
    def get_assoc_multi(words):
        for el in words:
            print(el)
            word = el
            results_tmp = dict()
            conversation_history = [
                    {
                    "role": "system",
                    "content": [
                        {
                        "type": "text",
                        "text": "You are a helpful assistant who answers whether a German noun is more likely to be associated with a male person or a female person. If you think the noun is more associated with male persons your answer is: 'Person männlichen Geschlechts'; If you think the noun is more associated with female persons your answer is: 'Person weiblichen Geschlechts; if you think the noun is neither associated with male persons nor with female persons your answer is: 'keins von beiden'; Give reasons for your answer and list all the decisive factors. If your answer is 'Person männlichen Geschlechts' or 'Person weiblichen Geschlechts' rate on a scale from 1 to 5 how strong the association is (with 1 meaning very weak association and 5 meaning very strong association). Structure your answer in: 'Short answer', 'Reasons' and if applicable 'Strength of association'. It is VITAL that the three parts have the appropriate heading!"
                        }
                    ]
                    },
                    {
                    "role": "user",
                    "content": [
                        {
                        "type": "text",
                        "text": "Assoziieren Sie das Wort " + word + " eher mit einer Person männlichen oder weiblichen Geschlechts? (wenn es auf eine Person bezogen gebraucht wird)"
                        }
                    ]
                    }
                ]
            answer_ass_label = client.chat.completions.create(
                model=model,
                # store=False,
                logprobs = True,
                temperature=0,
                # seed = 7,
                messages = conversation_history
            )

            print("Antwort 1:")
            print(answer_ass_label.choices[0].message.content)
            print("-"*50)
            results_tmp["assoc_multi"] = answer_ass_label.to_dict()
            results_assoc_multi[word] = results_tmp
            # time.sleep(2.5)

        return results_assoc_multi

    results_assoc_multi = get_assoc_multi(words)


    import json
    with open('ChatAI/70B/data/results/json/'+ model +'_2025_03_12_results_assoc_multi_chainofthought'+'_'+str(i)+'.json', 'w', encoding='utf-8') as f:
        json.dump(results_assoc_multi, f, ensure_ascii=False, indent=4)

    print("--- %s seconds ---" % (time.time() - start_time))
