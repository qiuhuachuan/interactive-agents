import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"
model_name = 'qiuhuachuan/simpsybot_Q'
simpsybot_qwen2_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
simpsybot_qwen2_tokenizer = AutoTokenizer.from_pretrained(model_name)

class Msg(BaseModel):
    messages: list

SYSTEM_PROMPT = """现在你是虚拟心理咨询师。
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
9. 更多的是引导用户思考和探索。"""



def get_prediction_simpsybot_qwen2(messages: list):
    system_item = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    messages = system_item + messages
    ctx = simpsybot_qwen2_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = simpsybot_qwen2_tokenizer([ctx], return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = simpsybot_qwen2_model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = simpsybot_qwen2_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/v1/chat/simpsybot_qwen2")
async def chat(msg: Msg):
    messages = msg.messages
    response = get_prediction_simpsybot_qwen2(messages=messages)
    return {'response': response}

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=9006)
    # CUDA_VISIBLE_DEVICES=7 nohup python -u simpsybot_Q.py > ./simpsybot_Q.log &
