import os
import re
import bs4
import torch
import streamlit as st
from typing import List

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# ============================================
# 0. 網頁設定 & CSS 美學 (PLUS ULTRA STYLE)
# ============================================
st.set_page_config(
    page_title="U.A. Database | 雄英資料庫",
    page_icon="🦸‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 注入 Google Fonts 和自定義 CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bangers&family=Noto+Sans+TC:wght@400;700&display=swap');

    /* 全局背景與字體 */
    .stApp {
        background-color: #f0f2f6;
        font-family: 'Noto Sans TC', sans-serif;
    }

    /* 標題特效 (美漫風格) */
    h1 {
        font-family: 'Bangers', cursive;
        color: #d32f2f;
        text-transform: uppercase;
        font-size: 3.5rem !important;
        text-shadow: 3px 3px 0px #FBC02D;
        letter-spacing: 2px;
        margin-bottom: 0px;
    }
    
    .subtitle {
        color: #1565C0;
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        border-bottom: 3px solid #FBC02D;
        display: inline-block;
        padding-bottom: 5px;
    }

    /* 側邊欄優化 (高科技深色風) */
    section[data-testid="stSidebar"] {
        background-color: #1a1a1a;
        color: white;
    }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #FBC02D !important; /* 金黃色標題 */
    }
    section[data-testid="stSidebar"] span {
        color: #e0e0e0 !important;
    }

    /* 聊天氣泡優化 */
    .stChatMessage {
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
        border: 2px solid transparent;
    }
    .stChatMessage:hover {
        transform: translateY(-2px);
    }

    /* AI (Assistant) 氣泡 - 歐爾麥特配色 */
    div[data-testid="chatAvatarIcon-assistant"] {
        background-color: #d32f2f !important;
    }
    div[data-testid="stChatMessage"]:nth-child(even) {
        background: linear-gradient(135deg, #ffffff 0%, #fff8e1 100%);
        border-left: 5px solid #d32f2f;
    }

    /* User 氣泡 - 雄英制服配色 */
    div[data-testid="chatAvatarIcon-user"] {
        background-color: #1565C0 !important;
    }
    div[data-testid="stChatMessage"]:nth-child(odd) {
        background: linear-gradient(135deg, #e3f2fd 0%, #ffffff 100%);
        border-right: 5px solid #1565C0;
    }

    /* 按鈕美化 */
    .stButton>button {
        background: linear-gradient(90deg, #d32f2f, #b71c1c);
        color: white;
        border-radius: 30px;
        border: none;
        font-family: 'Bangers', cursive;
        font-size: 1.2rem;
        letter-spacing: 1px;
        box-shadow: 0 4px 0 #7f0000; /* 立體感 */
        transition: all 0.1s;
    }
    .stButton>button:active {
        box-shadow: 0 0 0 #7f0000;
        transform: translateY(4px);
    }
    
    /* 檢索來源卡片 */
    .source-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #FBC02D;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        font-size: 0.9rem;
    }
    
    /* 隱藏預設選單 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================
# 1. 核心函數 (保持你的邏輯不變)
# ============================================

@st.cache_resource
def get_device():
    if torch.cuda.is_available(): return "cuda"
    return "cpu"

@st.cache_resource
def get_embeddings(device):
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

def split_by_character_and_inject_context(text: str) -> List[Document]:
    text = re.sub(r'\[編輯\]|\[edit\]|\[\d+\]', '', text)
    character_pattern = r'\n(?=[\u4e00-\u9fa5]{2,}[（\(][^）\)]+[）\)])'
    sections = re.split(character_pattern, text)
    
    final_docs = []
    sub_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    for section in sections:
        section = section.strip()
        if not section: continue
        lines = section.split('\n')
        try:
            char_header = lines[0].split('／')[0].split('（')[0].strip()
        except:
            char_header = "機密檔案"
        
        for chunk in sub_splitter.split_text(section):
            final_docs.append(Document(
                page_content=f"【英雄檔案: {char_header}】\n{chunk}",
                metadata={"character": char_header}
            ))
    return final_docs

@st.cache_resource
def initialize_vectorstore(_embeddings):
    index_path = "./faiss_index_hero_v3"
    
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, _embeddings, allow_dangerous_deserialization=True)
    
    with st.status("📡 正在連接雄英伺服器...", expanded=True) as status:
        st.write("正在從維基百科提取英雄數據...")
        url = "https://zh.wikipedia.org/wiki/%E6%88%91%E7%9A%84%E8%8B%B1%E9%9B%84%E5%AD%A6%E9%99%A2%E8%A7%92%E8%89%B2%E5%88%97%E8%A1%A8"
        loader = WebBaseLoader(url, bs_kwargs=dict(parse_only=bs4.SoupStrainer(id="mw-content-text")))
        raw_docs = loader.load()
        
        st.write("正在進行數據向量化...")
        splits = split_by_character_and_inject_context(raw_docs[0].page_content)
        vectorstore = FAISS.from_documents(splits, _embeddings)
        vectorstore.save_local(index_path)
        status.update(label="✅ 資料庫同步完成！", state="complete", expanded=False)
        return vectorstore

# ============================================
# 2. 側邊欄 (Hero Support Item Interface)
# ============================================
with st.sidebar:
    st.image("assets/My_Hero_Academia_logo.png", use_container_width=True)
    st.markdown("<div style='text-align: center; color: #aaa; margin-bottom: 20px;'>SECURE TERMINAL V.3.1</div>", unsafe_allow_html=True)
    
    device = get_device()
    st.markdown(f"**🟢 運算核心狀態:** `{device.upper()}`")
    if device == "cuda":
        st.caption(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    
    st.markdown("---")
    st.markdown("### 🛠️ 參數設定")
    
    model_name = st.selectbox("Model", ["deepseek-r1:8b", "llama3"], index=0)
    temperature = st.slider("思維發散度 (Temp)", 0.0, 1.0, 0.2)
    k_retrieval = st.slider("資料調閱權限 (Docs)", 1, 10, 4)
    
    st.markdown("---")
    col_reset, col_help = st.columns(2)
    with col_reset:
        if st.button("🔄 重啟系統"):
            st.session_state.messages = []
            st.rerun()

# ============================================
# 3. 主介面 (Hero Interface)
# ============================================

# 初始化
if "messages" not in st.session_state:
    st.session_state.messages = []

# Header 區域
col_logo, col_title = st.columns([1, 6])
with col_logo:
    # 這裡可以用 font-awesome 或者真正的圖片
    st.markdown("<div style='font-size: 60px; text-align: center;'>🦸‍♂️</div>", unsafe_allow_html=True)
with col_title:
    st.markdown("<h1>U.A. HIGH DATABASE</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>我的英雄學院英雄知識問答系統 / PLUS ULTRA !!</div>", unsafe_allow_html=True)

# 載入系統
embeddings = get_embeddings(device)
vectorstore = initialize_vectorstore(embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": k_retrieval})

# 顯示對話歷史
for message in st.session_state.messages:
    avatar = "🧑‍🎓" if message["role"] == "user" else "🦸‍♂️"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# 輸入區
if query := st.chat_input("請輸入問題"):
    
    # 1. 使用者輸入
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(query)

    # 2. 檢索過程 (使用折疊選單，保持介面整潔)
    with st.status("🔍 正在檢索機密檔案...", expanded=False) as status:
        docs = retriever.invoke(query)
        context_text = "\n\n".join([d.page_content for d in docs])
        
        # 顯示漂亮的來源卡片
        st.markdown("### 📂 檢索到的檔案片段")
        for i, doc in enumerate(docs):
            char_name = doc.metadata.get('character', '未知')
            content_preview = doc.page_content.replace(f"【英雄檔案: {char_name}】", "").strip()[:100]
            st.markdown(f"""
            <div class="source-card">
                <b>#{i+1} 檔案來源: {char_name}</b><br>
                <span style="color: #666;">{content_preview}...</span>
            </div>
            """, unsafe_allow_html=True)
        status.update(label=f"✅ 檢索完成！共發現 {len(docs)} 筆關聯資料", state="complete")

    # 3. AI 回答
    with st.chat_message("assistant", avatar="🦸‍♂️"):
        response_placeholder = st.empty()
        
        # Prompt
        llm = ChatOllama(model=model_name, temperature=temperature)
        prompt_template = ChatPromptTemplate.from_template(

            """請根據提供的資料回答問題。
            資料：{context}
            問題：{question}
            回答：
            回答規則：
            1. 條列式呈現重點。
            2. 使用繁體中文。
            """
        )
        
        chain = (
            {"context": lambda x: context_text, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
        )
        
        # 串流輸出 + DeepSeek 思考分離
        full_response = ""
        final_answer_buffer = ""
        thought_buffer = ""
        is_thinking = False
        
        # 建立思考區塊 (如果模型支援)
        thought_expander = None 
        
        try:
            for chunk in chain.stream(query):
                full_response += chunk
                
                # 處理 <think> 標籤
                if "<think>" in chunk:
                    is_thinking = True
                    thought_expander = st.expander("🧠 戰術分析過程 (DeepSeek)", expanded=True)
                    chunk = chunk.replace("<think>", "")
                
                if "</think>" in chunk:
                    is_thinking = False
                    chunk = chunk.replace("</think>", "")
                    
                if is_thinking and thought_expander:
                    thought_buffer += chunk
                    thought_expander.markdown(f"_{thought_buffer}_")
                else:
                    final_answer_buffer += chunk
                    response_placeholder.markdown(final_answer_buffer + "▌")
            
            # 最後顯示完整文字 (移除游標)
            response_placeholder.markdown(final_answer_buffer)
            
        except Exception as e:
            st.error(f"❌ 系統錯誤: 請確認 Ollama 是否已啟動 ({e})")

    # 4. 存檔
    st.session_state.messages.append({"role": "assistant", "content": final_answer_buffer})