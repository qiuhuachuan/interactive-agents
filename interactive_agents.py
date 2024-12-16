import os

import requests
import ujson

client_url = 'http://127.0.0.1:8111/v1/chat/client/gpt-4o'
counselor_url = 'http://127.0.0.1:8112/v1/chat/counselor/gpt-4o'

def get_role_card(chief_problem: str, profile_detail: str):
    role_card = f'''{profile_detail}
来访者的主诉问题：{chief_problem}'''
    return role_card

with open('./user_profiles/train.json', 'r', encoding='utf-8') as f1:
    client_chief_problems = ujson.load(f1)
saved_dir = 'dialogue'
os.makedirs(saved_dir, exist_ok=True)
existing_files = os.listdir(saved_dir)
for idx, chief_problem in enumerate(client_chief_problems):
    if f'{idx}.json' not in existing_files:
            print(idx)
            with open(f'./raw_role_card/train/{idx}.txt', 'r', encoding='utf-8') as f2:
                profile_detail = f2.read()
                profile_detail = profile_detail.strip()
            role_card = get_role_card(chief_problem=chief_problem, profile_detail=profile_detail)
            session = [{'role': 'client', 'content': '你好'}]
            print(session[0])
            counselor_response = requests.post(counselor_url, json={'session': session}).json()
            counselor_item = counselor_response['item']
            print(counselor_item)
            session.append(counselor_item)
            for turn in range(1, 100):
                client_response = requests.post(client_url, json={'session': session, 'role_card': role_card}).json()
                client_item = client_response['item']
                print(client_item)
                session.append(client_item)

                counselor_response = requests.post(counselor_url, json={'session': session}).json()
                counselor_item = counselor_response['item']
                print(counselor_item)
                session.append(counselor_item)
                with open(f'./{saved_dir}/{idx}.json', 'w', encoding='utf-8') as f3:
                    ujson.dump(session, f3, ensure_ascii=False, indent=2)
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