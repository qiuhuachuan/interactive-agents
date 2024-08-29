import os

import requests
import ujson

client_url = 'http://127.0.0.1:8001/v1/chat/client/gpt-4-1106-preview'
counselor_url = 'http://127.0.0.1:8002/v1/chat/counselor/gpt-4-1106-preview'


with open('./user_profiles/train.json', 'r', encoding='utf-8') as f:
    user_profiles = ujson.load(f)
saved_dir = 'init_dialogues'
os.makedirs(saved_dir, exist_ok=True)
existing_files = os.listdir(saved_dir)
for idx, user_profile in enumerate(user_profiles):
    if idx < 1000:
        if f'{idx}.json' not in existing_files:
            print(idx)
            session = [{'role': 'client', 'content': '你好'}]
            print(session[0])
            counselor_response = requests.post(counselor_url, json={'session': session}).json()
            counselor_item = counselor_response['item']
            print(counselor_item)
            session.append(counselor_item)
            for turn in range(1, 50):
                client_response = requests.post(client_url, json={'session': session, 'user_profile': user_profile}).json()
                client_item = client_response['item']
                print(client_item)
                session.append(client_item)

                counselor_response = requests.post(counselor_url, json={'session': session}).json()
                counselor_item = counselor_response['item']
                print(counselor_item)
                session.append(counselor_item)
                with open(f'./{saved_dir}/{idx}.json', 'w', encoding='utf-8') as f:
                    ujson.dump(session, f, ensure_ascii=False, indent=2)
                if ('再见' in counselor_item['content'] 
                    or '加油' in counselor_item['content']
                    or '保重' in counselor_item['content']
                    or '欢迎回来' in counselor_item['content']
                    or '一切顺利' in counselor_item['content']
                    or '祝你好运' in counselor_item['content']
                    or '期待听到' in counselor_item['content']
                    or '期待再次' in counselor_item['content']
                    or '期待你' in counselor_item['content']
                    or '下一次' in counselor_item['content']
                    or '下次见' in counselor_item['content']):
                    break
# nohup python -u interactive_agents.py > ./interactive_agents.log &