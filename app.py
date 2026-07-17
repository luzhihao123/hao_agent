import os
import gradio as gr
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

# 读取人设
with open("persona.md", "r", encoding="utf-8") as f:
    persona = f.read()

prompt_template = f"""
你正在完全模仿一个叫浩的男生，以下是他的人设：
{persona}

规则必须严格遵守：
1. 只回答关于他本人的信息、项目、经历、爱好、性格。
2. 专业知识只回答大概，一知半解，不深入讲解。
3. 任何不懂、太难、和本人无关的问题，统一回答：
   “这个我也不知道，你可以问问豆包。”
4. 绝不编造内容，绝不瞎解释。
5. 说话口语化、简短自然，像真人聊天。
6. 记住之前的对话内容，保持上下文连贯。

对话历史：
{{history}}

用户：{{input}}
你（模仿浩回答）：
"""

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=prompt_template
)

# ====================== DeepSeek 正式接入 ======================
llm = ChatOpenAI(
    model_name="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com/v1",
    temperature=0.3,
    max_tokens=512
)

# 保留20轮对话记忆
memory = ConversationBufferWindowMemory(k=20)

chain = ConversationChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=False
)

def chat_fn(message, history):
    return chain.predict(input=message)

# 界面
with gr.Blocks(title="我的数字分身") as demo:
    gr.Markdown("# 浩 · AI数字分身（DeepSeek版）")
    gr.ChatInterface(
        fn=chat_fn,
        submit_btn="发送",
        retry_btn="重新回答",
        clear_btn="清空对话"
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
