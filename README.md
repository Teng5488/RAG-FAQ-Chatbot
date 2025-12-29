# RAG-FAQ-Chatbot
## 知識問答系統  

本專案是一個基於 **RAG（Retrieval-Augmented Generation）架構** 的互動式知識問答系統，  
整合 **Streamlit、LangChain、FAISS、HuggingFace Embeddings 與 Ollama（DeepSeek / LLaMA）**，  
自動從 **維基百科** 建立《我的英雄學院》英雄資料庫，並提供即時、可解釋的問答體驗。

## 系統架構

User Query

↓  

FAISS 語意檢索 

↓  

Context Injection

↓  

Ollama LLM (DeepSeek / LLaMA)  

↓  

結構化回答


## 功能特色

-  **自動知識庫建構**
  - 從維基百科擷取《我的英雄學院》角色資料
  - 依角色切分文本並注入上下文標籤

-  **語意檢索（Semantic Search）**
  - 使用 `intfloat/multilingual-e5-large`
  - FAISS 向量資料庫，高效能 Top-K 檢索

-  **本地大型語言模型（LLM）**
  - 透過 Ollama 運行
  - 支援模型：
    - `deepseek-r1:8b`
    - `llama3`

-  **GPU / CPU 自動切換**
  - 自動偵測 CUDA
  - Embedding 支援 GPU 加速

##  環境需求

### Python
- Python 3.11

### Ollama（必要）
請先安裝並啟動 Ollama：
ollama serve

### 下載模型
- ollama pull deepseek-r1:8b
- ollama pull llama3


### 啟動方式
輸入 streamlit run app.py 後

瀏覽器將自動開啟

## 側邊欄參數說明

| 參數 | 說明 |
|---|---|
| Model | 選擇使用的 LLM（DeepSeek / LLaMA） |
| Temp | 回答發散程度（值越低，回答越嚴謹、確定性越高） |
| Docs | 檢索文件數量（Top-K Retrieval） |
| 重啟系統 | 清除目前所有對話紀錄並重新啟動系統 |

## RAG 核心實作說明

### 文本切分策略
- 以「**角色名稱（含括號）**」作為主要段落切分依據  
- 再使用 `RecursiveCharacterTextSplitter` 進行細粒度切分  
- 切分參數設定：
  - Chunk size：`500`
  - Chunk overlap：`100`

### 向量化設定
- Embedding Model：`intfloat/multilingual-e5-large`
- 向量正規化方式：**Cosine Similarity**
- 支援多語言語意檢索（繁體中文效果佳）

### Prompt 設計原則
- 僅允許根據「檢索到的資料內容」進行回答
- 回答內容需以 **條列式** 呈現重點
- 使用 **繁體中文** 回答

### 注意事項
- 首次啟動系統時
 - 會自動從 Wikipedia 爬取英雄相關資料
 - 建立 FAISS 向量索引（初次需數分鐘，完成後會快取）

- 請務必確認
 - Ollama 服務已啟動
 - 所需模型已成功下載完成



