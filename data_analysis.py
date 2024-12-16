import numpy as np
import ujson

turn_counter = []
utter_counter_all = 0
utter_counter_client = 0
utter_counter_counselor = 0
utter_length_client = 0
utter_length_counselor = 0

for idx in range(1000):
    with open(f'./dialogue/{idx}.json', 'r', encoding='utf-8') as f:
        dialogue = ujson.load(f)
    for item in dialogue:
        role = item['role']
        content = item['content']
        if role == 'user':
            utter_counter_client += 1
            utter_length_client += len(content)
        else:
            assert role == 'assistant'
            utter_counter_counselor += 1
            utter_length_counselor += len(content)
    turn_counter.append(len(dialogue)/2)

print(np.mean(turn_counter)) # 12.948
print(np.max(turn_counter)) # 25.0
print(np.min(turn_counter)) # 7.0
print(utter_counter_all) # 25862
print(utter_counter_client) # 12948
print(utter_counter_counselor) # 1248
print(utter_length_client/utter_counter_client) # 54.1
print(utter_length_counselor/utter_counter_counselor) # 70.8

