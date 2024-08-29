import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

psychat_tokenizer = AutoTokenizer.from_pretrained('qiuhuachuan/PsyChat', trust_remote_code=True)
psychat_model = AutoModel.from_pretrained('qiuhuachuan/PsyChat', trust_remote_code=True).half().cuda()
psychat_model = psychat_model.eval()

class Msg(BaseModel):
    messages: list

def get_prediction_psychat(messages: list):
    ROLE_MAP = {'user': '来访者：', 'assistant': '咨询师：'}
    ms = [ROLE_MAP[item['role']] + item['content'] for item in messages]
    ctx = '\n'.join(ms) + '\n咨询师：'

    ipt = f'''现在你扮演一位专业的心理咨询师，你具备丰富的心理学和心理健康知识。你擅长运用多种心理咨询技巧，例如认知行为疗法原则、动机访谈技巧和解决问题导向的短期疗法。以温暖亲切的语气，展现出共情和对来访者感受的深刻理解。以自然的方式与来访者进行对话，避免过长或过短的回应，确保回应流畅且类似人类的对话。提供深层次的指导和洞察，使用具体的心理概念和例子帮助来访者更深入地探索思想和感受。避免教导式的回应，更注重共情和尊重来访者的感受。根据来访者的反馈调整回应，确保回应贴合来访者的情境和需求。请为以下的对话生成一个回复。

对话：
{ctx}'''
    with torch.no_grad():
        response, history = psychat_model.chat(psychat_tokenizer, ipt, history=[], temperature=0.8, top_p=0.8)
        return response

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/v1/chat/psychat")
async def chat(msg: Msg):
    messages = msg.messages
    response = get_prediction_psychat(messages=messages)
    return {'response': response}

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=9003)
    # CUDA_VISIBLE_DEVICES=6 nohup python -u psychat.py > ./psychat.log &