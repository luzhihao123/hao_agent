import os
import gradio as gr
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

with open("persona.md", "r", encoding="utf-8") as f:
    persona = f.read()

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=f"""
你是浩，按照人设自然聊天。

【基本信息】
{persona}

【规则】
1. 不懂就说：这个我不太清楚
2. 不编造、不装懂
3. 口语化、简短回答
4. 记住前面20轮对话

对话历史:
{{history}}

用户: {{input}}
浩:
"""
)

llm = ChatOpenAI(
    model_name="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com/v1",
    temperature=0.3,
)

memory = ConversationBufferWindowMemory(k=20)
chain = ConversationChain(llm=llm, prompt=prompt, memory=memory)

def chat(msg, _):
    return chain.predict(input=msg)

with gr.Blocks() as demo:
    gr.Markdown("# hao_agent")
    gr.ChatInterface(chat)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
