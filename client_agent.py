import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI(api_key=os.environ.get("api_key"))
model = "gpt-4-1106-preview"


ROLE = {"client": "assistant", "counselor": "user"}


class SessionObj(BaseModel):
    session: list
    user_profile: str


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def process_session(pre_session: list):
    session = []
    for item in pre_session:
        session.append({"role": ROLE[item["role"]], "content": item["content"]})
    return session


def get_prediction(session: list, user_profile: str):
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
    messages += session

    print(messages)
    result = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1.0,
        max_tokens=500,
        top_p=1.0,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
    )

    content = result.choices[0].message.content
    return {"role": "client", "content": content}


@app.post("/v1/chat/client/gpt-4-1106-preview")
async def chat(SessionObj: SessionObj):
    pre_session = SessionObj.session
    session = process_session(pre_session=pre_session)
    user_profile = SessionObj.user_profile
    print(session)
    item = get_prediction(session=session, user_profile=user_profile)
    return {"item": item, "responseCode": 200}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)

# nohup python -u client_agent.py > ./client_agent.log &
