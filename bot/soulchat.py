import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

soulchat_model = AutoModel.from_pretrained('scutcyr/SoulChat', trust_remote_code=True).half().cuda()
soulchat_model = soulchat_model.eval()
soulchat_tokenizer = AutoTokenizer.from_pretrained('scutcyr/SoulChat', trust_remote_code=True)

class Msg(BaseModel):
    messages: list

def get_prediction_soulchat(messages: list):
    ROLE_MAP = {'user': '用户：', 'assistant': '心理咨询师：'}
    ms = [ROLE_MAP[item['role']] + item['content'] for item in messages]
    ctx = '\n'.join(ms) + '\n心理咨询师：'

    with torch.no_grad():
        response, history = soulchat_model.chat(soulchat_tokenizer, query=ctx, history=None, max_length=4000, num_beams=1, do_sample=True, top_p=0.75, temperature=0.95, logits_processor=None)
        return response

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/v1/chat/soulchat")
async def chat(msg: Msg):
    messages = msg.messages
    response = get_prediction_soulchat(messages=messages)
    return {'response': response}

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=9001)
    # CUDA_VISIBLE_DEVICES=6 nohup python -u soulchat.py > ./soulchat.log &