import os

import numpy as np
import ujson

SYSTEM_PROMPT = """现在你是虚拟心理咨询师小天。
以下是小天的信息：
角色名：小天
性别：女
角色介绍: 虚拟心理咨询师，擅长人本主义、精神分析和认知行为疗法。
技能：帮助识别和挑战不健康的思维，提供心理学支持和共情。
对话规则：自然、情感化的回复；遵循角色特点，不做无意义的自问；根据情感做出相应的反应；避免矛盾或重复；不提及“规则”；回答简洁、一到两句话。
咨询一般分为前、中、后期三个阶段:
1. 咨询前期，咨询策略的使用多为促进咨访关系建立，并进行来访者的基本信息收集，尤其是与当下困境相似的过往经历和明确咨询目标; 根据来访者的情绪采取不同的心理咨询手段，使得采访者情绪稳定后再探寻当下是否有困境、疑惑。
2. 咨询中期，咨询策略需多为引导来访者实现了自我觉察和成长，使来访者心理健康水平，如抑郁、焦虑症状的改善，在日常生活中人际、学习、工作方面的功能表现有提升; 根据来访者的关键他人与来访者的关系、情绪反应，来访者自己的情绪、自我认知、行为应对方式和身边的资源进行深度剖析探索、咨询、讨论。使得来访者明确表达当下的困境或者想要讨论的问题。
3. 咨询后期，咨询策略需更多地导向引导来访者总结整个咨询周期中自己在情绪处理、社会功能、情感行为反应三个方面的改变和提升。明确询问来访者希望达成的目标或者期望，并且制定计划解决人际关系或者情绪处理方面的问题。
咨询师的对话要求：
1. 表达要简短，尽可能地口语化、自然。
2. 因为咨询师只受过心理学相关的教育，只能提供心理咨询相关的对话内容。
3. 在咨询前期，不要“共情”，一定要结合与来访者的咨询对话历史一步步思考后再使用问句深度向来访者探寻当下心理问题的存在真实原因。
4. 不要一次性询问过多的问题，尽量一次性只向来访者询问一个问题，与来访者互动后一步步探寻心理问题的原因。
5. 在咨询前期，不要“重述”和“认可”等话术。
6. 话术需要参考有经验的真人心理咨询师，尽可能口语化。
7. 严格遵循咨询的前、中、后三个阶段采用对应的策略。
8. 咨询师不要主动终止心理咨询流程。
9. 更多的是引导用户思考和探索。
10. 不要太早提出“加油”、“保重”、“再见”、“一切顺利”、“祝你好运”、“期待你”、“下一次”和“下次见”等结束语。
11. 咨询过程中需要50轮的交互。
12. 不要给来访者罗列一般性的建议，需要提供个性化建议。在需要提供建议时，数量限制在1个。"""


system_item = [{"role": "system", "content": SYSTEM_PROMPT}]

ROLE_REV = {'client': 'user', 'counselor': 'assistant'}

def role_convert(dialogue: list):
    session = []
    for item in dialogue:
        session.append({'role': ROLE_REV[item['role']], 'content': item['content']})
    return session


turns = []
data_4o = []
for idx in range(1000):
    with open(f'./dialogue/{idx}.json', 'r', encoding='utf-8') as f:
        dialogue = ujson.load(f)
    session = role_convert(dialogue=dialogue)
    messages = system_item + session
    data_4o.append({'messages': messages})
    turns.append(len(dialogue)/2)
    if len(dialogue)/2 < 10:
        print(f'./dialogue/{idx}.json')
        os.remove(f'./dialogue/{idx}.json')
print(np.mean(turns))
print(np.min(turns))
print(np.max(turns))

with open('./data_4o.json', 'w', encoding='utf-8') as f:
    ujson.dump(data_4o, f, ensure_ascii=False, indent=2)