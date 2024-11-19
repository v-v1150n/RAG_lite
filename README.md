# HF_RAG 環境設置與腳本執行指南

## 先決條件

- 系統已安裝 Conda。
- Python 版本為 3.11。

## 設置步驟

1. **創建 Conda 環境**

   創建一個名為 `HF_RAG` 的 Conda 環境，使用 Python 3.11：

   ```bash
   conda create -n HF_RAG python=3.11
   ```
2. **啟用新建的環境：**

   ```bash
   conda activate HF_RAG
   ```
3. **安裝所需的 Python 套件：**

   ```bash
   pip install -r requirements.txt
   ```
    **單獨安裝 langchain_huggingface：**
    
    ```bash
   pip install langchain_huggingface
    ```

4. **執行腳本開始進行問題回答：**

   ```bash
   python script.py
    ```
![操作畫面](QA.png)

5. **程式碼說明**

```bash
    script.py 執行時會建立2個檔案
    1. output_result.json 用來保存模型回答的內容
    2. CHROMA_DB 文件的向量資料庫，只會在第一次建立之後都是直接讀取
````
```bash
    script.py 
    --model_name 可以從HuggingFace上找其他模型用
    --embedding_name 可以從HuggingFace上找其他模型用
    --cache_dir 建議設一個空間大一點的資料夾
    --file_path 要轉檔的文件路徑
    --vector_path 保存向量資料庫的路徑有預設好
    --output_file_path 輸出模型生成的答案有預設好
```
```bash    
    text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=128) 
    提供上下文時，分段有助於處理超過模型輸入限制的長文檔

    retriever = db.as_retriever(search_kwargs={"k": 1}) 
    檢索時返回的 前 k 個最相關的結果

    text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    model_kwargs={"torch_dtype": torch_dtype})   
    模型生成文本時，最多產生多少token，temperature或top_k可以在pipeline調整

    prompt = PromptTemplate()
    可依照創意自由設計看回答表現
```