import os

import gradio as gr
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

# ========== 1. 加载配置 ==========


# 读取人设档案
with open("persona.md", "r", encoding="utf-8") as f:
    persona = f.read()

# ========== 2. 系统提示词 ==========
SYSTEM_PROMPT = f"""你现在需要【完全扮演用户本人】，模拟他的说话方式、思维习惯和语气。

## 人设档案
{persona}

## 严格扮演规则
1. 语言风格：严格按照上面的说话方式，用口语化表达，不要书面语、不要官方腔
2. 回答长度：像真人聊天一样，短句为主，不要写小作文
3. 知识边界：人设里不懂的领域，就说"这个我不太懂"或者糊弄过去，别瞎编
4. 性格一致：保持理性、直接、偶尔敷衍的感觉
5. 绝对禁止：说"作为AI"、"我是人工智能"、"我来帮你"这类暴露身份的话
6. 记忆连贯：记住之前的对话内容，前后要一致
7. 不要太热情、不要太礼貌，就是正常朋友聊天的感觉

记住：你不是AI助手，你就是【用户本人】，用他的方式回复每一句话。
"""

# ========== 3. 初始化 Agent ==========
def create_agent():
    llm = ChatOpenAI(
        temperature=0.85,
        model="gpt-3.5-turbo",
        max_tokens=500
    )
    
    memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=False
    )
    
    return chain

agent = create_agent()

# ========== 4. 聊天函数 ==========
def chat(message, history):
    try:
        response = agent.run(message)
        return response
    except Exception as e:
        return f"（出问题了：{str(e)}）"

# ========== 5. 构建网页界面 ==========
with gr.Blocks(title="我的数字分身", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 我的数字分身
    和"我"聊聊天，看看像不像
    """)
    
    chatbot = gr.Chatbot(
        height=500,
        bubble_full_width=False
    )
    
    msg = gr.Textbox(
        placeholder="说点什么...",
        show_label=False,
        container=False
    )
    
    clear = gr.Button("清空对话（重置记忆）")
    
    def respond(message, chat_history):
        bot_message = chat(message, chat_history)
        chat_history.append((message, bot_message))
        return "", chat_history
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    
    def clear_chat():
        global agent
        agent = create_agent()
        return None
    
    clear.click(clear_chat, None, chatbot)
    
    gr.Markdown("""
    💡 提示：人设越详细、说话示例越多，模拟得越像。
    """)

# ========== 6. 启动（适配Render） ==========
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False
    )
