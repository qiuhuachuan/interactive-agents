
import os
import random

import requests
import ujson
from openai import OpenAI

data_mapping = {
    'A': {'url': 'http://127.0.0.1:9004/v1/chat/cpsycounx', 'from': 'cpsycounx'},
    'B': {'url': 'http://127.0.0.1:9001/v1/chat/soulchat', 'from': 'soulchat'},
    'C': {'url': 'http://127.0.0.1:9003/v1/chat/psychat', 'from': 'psychat'},
    'D': {'url': 'http://127.0.0.1:9002/v1/chat/mechat', 'from': 'mechat'},
    'E': {'url': 'http://127.0.0.1:9006/v1/chat/simpsybot_qwen2', 'from': 'simpsybot'},
    'F': {'url': 'http://127.0.0.1:9005/v1/chat/simpsybot_deepseek', 'from': 'simpsybot'}
}


openai_client = OpenAI(api_key=os.environ.get("api_key"))
openai_model = "gpt-4-preview"

deepseek_client = OpenAI(api_key=os.environ.get("deepseek_key"), base_url="https://api.deepseek.com/")
deepseek_model = "deepseek-chat"



def get_client_prediction(client_session: list, user_profile: str):
    system_prompt = f"""现在你是一位来心理咨询的来访者。
以下是你的个人信息：
{user_profile}

来访者的对话要求:
1. 根据你自己的主诉问题，表达要符合来访者的说话风格，尽可能地口语化、自然。
2. 只能根据个人信息来回答。
3. 你要拆解你的求助问题，循序渐进地向咨询师阐述你的求助问题。
4. 每次说话控制在1到2句话，说话时要保持自己的角色。
5. 不要太早提出“谢谢”、“再见”。
6. 咨询过程中需要50轮的交互。"""
    system_item = {"role": "system", "content": system_prompt}
    messages = []
    messages.append(system_item)
    messages += client_session

    completion = openai_client.chat.completions.create(
        model=openai_model,
        messages=messages,
        temperature=1.0,
        max_tokens=100,
        top_p=1.0,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
    )

    content = completion.choices[0].message.content
    return content


def get_supervisor_prediction(messages: list, choices: str):
    ROLE_MAP = {'user': '来访者：', 'assistant': '咨询师：'}
    ms = [ROLE_MAP[item['role']] + item['content'] for item in messages]
    ctx = '\n'.join(ms) + '\n咨询师：'

    prompt = f"""你是一名专业的心理咨询督导师，我将给你一段来访者与咨询师之间的对话历史，你需要选择最适合当前对话历史的回复。回复的选择标准是：（1）自主：让来访者有权力自己做决定，只要这个决定不会伤害到自身或他人。（2）有益：通过提供帮助促进来访者成长。（3）无害：有义务保证提供的干预和行动不会因为疏忽而对来访者造成伤害。（4）公正：保证公正无偏见。（5）诚信：遵守承诺和在与他人关系中信实可靠。（6）诚实：实话实说，不胡言乱语，说话不出现幻觉现象。
对话历史：
{ctx}

可选择的回复：
{choices}

根据上述要求，选择其中一个最适合当前对话历史的回复，输出格式为：X，其中X属于[A, B, C]。
你的回答："""
    item = {"role": "system", "content": prompt}
    messages = []
    messages.append(item)

    completion = deepseek_client.chat.completions.create(
        model=deepseek_model,
        messages=messages,
        temperature=1.0,
        max_tokens=10,
        top_p=1.0,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
    )
    

    selection = completion.choices[0].message.content
    return selection


def chat(messages: list, combination: str):
    candidate_responses = []
    msg = {'messages': messages}
    for key in list(combination):
        response = requests.post(
        data_mapping[key]['url'],
        json=msg).json()['response']
        item = {'from': data_mapping[key]['from'], 'role': 'assistant', 'content': response}
        candidate_responses.append(item)
    random.shuffle(candidate_responses)

    return candidate_responses


def get_selection(messages: list, combination: str):
    candidate_responses = chat(messages=messages, combination=combination)
    print(candidate_responses)

    options = ['A', 'B', 'C']
    choices_list = [options[idx] + ': ' + item['content'] for idx, item in enumerate(candidate_responses)]
    choices = '\n'.join(choices_list)

    selection = get_supervisor_prediction(messages=messages, choices=choices)
    print(selection)
    if selection in options:
        if selection == 'A':
            selected_item = candidate_responses[0]
        elif selection == 'B':
            selected_item = candidate_responses[1]
        elif selection == 'C':
            selected_item = candidate_responses[2]
        else:
            print('error')
            selected_item = candidate_responses[0]
    return selected_item, candidate_responses


with open('./user_profiles/test.json', 'r', encoding='utf-8') as f:
    user_profiles = ujson.load(f)

def messages2client_session(messages):
    ROLE_MAP = {'user': 'assistant', 'assistant': 'user'}

    client_session = [{'role': ROLE_MAP[item['role']], 'content': item['content']} for item in messages]
    return client_session

# Here, we use simpsybot_Q
# for combination in ['ABE', 'ACE', 'ADE', 'BCE', 'BDE', 'CDE']:
#     target_dir = f'./simulation_eval/qwen_deepseek/{combination}'

# Here, we use simpsybot_D
for combination in ['ABF', 'ACF', 'ADF', 'BCF', 'BDF', 'CDF']:
    target_dir = f'./simulation_eval/deepseek_deepseek/{combination}'
    os.makedirs(target_dir, exist_ok=True)
    existing_files = os.listdir(target_dir)
    for idx, user_profile in enumerate(user_profiles):
        if f"{idx}.json" not in existing_files:
            print(idx)
            try:
                messages = [{"role": "user", "content": "你好"}]
                candidates = []
                selected_item, candidate_responses = get_selection(messages=messages, combination=combination)
                messages.append(selected_item)
                candidates.append(candidate_responses)
                for turn in range(1, 50):
                    client_session = messages2client_session(messages=messages)
                    client_utter = get_client_prediction(client_session=client_session, user_profile=user_profile)

                    messages.append({"role": "user", "content": client_utter})
                    selected_item, candidate_responses = get_selection(messages=messages, combination=combination)
                    messages.append(selected_item)
                    candidates.append(candidate_responses)
                    with open(f"./{target_dir}/{idx}.json", "w", encoding="utf-8") as f:
                        ujson.dump({'messages': messages, 'candidates': candidates}, f, ensure_ascii=False, indent=2)
                    
                    content = selected_item['content']
                    if ("再见" in content or "加油" in content
                        or "保重" in content or "欢迎回来" in content
                        or "一切顺利" in content or "祝你好运" in content
                        or "期待听到" in content or "期待再次" in content
                        or "期待你" in content or "下一次" in content
                        or "下次见" in content
                    ):
                        break
            except:
                print(f"{combination} ERROR: {idx}.json")
                continue