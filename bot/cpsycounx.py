import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

cpsycounx_tokenizer = AutoTokenizer.from_pretrained('CAS-SIAT-XinHai/CPsyCounX', trust_remote_code=True)
cpsycounx_model = AutoModelForCausalLM.from_pretrained('CAS-SIAT-XinHai/CPsyCounX', torch_dtype=torch.float16, trust_remote_code=True).cuda()
cpsycounx_model = cpsycounx_model.eval()

class Msg(BaseModel):
    messages: list

def get_prediction_cpsycounx(messages: list):
    ROLE_MAP = {'user': '来访者：', 'assistant': '心理咨询师：'}
    ms = [ROLE_MAP[item['role']] + item['content'] for item in messages]
    ctx = '\n'.join(ms) + '\n咨询师：'

    with torch.no_grad():
        response, history = cpsycounx_model.chat(cpsycounx_tokenizer, query=ctx, history=[])
        return response

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/v1/chat/cpsycounx")
async def chat(msg: Msg):
    messages = msg.messages
    response = get_prediction_cpsycounx(messages=messages)
    return {'response': response}

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=9004)
    # CUDA_VISIBLE_DEVICES=6 nohup python -u cpsycounx.py > ./cpsycounx.log &