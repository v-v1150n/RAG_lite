import argparse
from langchain_chroma import Chroma  # 向量存儲
from langchain_community.document_loaders import PyPDFLoader  # 加載 PDF 文件
from langchain.text_splitter import CharacterTextSplitter  # 文本切分
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline  # 嵌入生成與管道
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, pipeline
)  # 模型和分詞器
from accelerate import Accelerator  # 加速器
from langchain_core.prompts import PromptTemplate  # 提示模板
from langchain_core.runnables import RunnablePassthrough  # 可執行工具
from langchain_core.output_parsers import StrOutputParser  # 字符串輸出解析器
import torch
import json
import os

# 定義命令行參數
parser = argparse.ArgumentParser(description="基於 LangChain 和 Hugging Face 模型的問答系統")
parser.add_argument("--cache_dir", default="/data/vv1150n/hugginface", help="模型緩存目錄")
parser.add_argument("--model_name", default="yentinglin/Llama-3-Taiwan-8B-Instruct", help="語言模型名稱")
parser.add_argument("--embedding_name", default="intfloat/multilingual-e5-large", help="嵌入模型名稱")
parser.add_argument("--file_path", default="AD.pdf", help="PDF 文件路徑")
parser.add_argument("--vector_path", default="./CHROMA_DB", help="向量資料庫路徑")
parser.add_argument("--output_file_path", default="output_result.json", help="輸出文件路徑")

args = parser.parse_args()
print("Initializing...")

# 初始化輸出文件，若不存在則創建為空列表
if not os.path.exists(args.output_file_path):
    with open(args.output_file_path, "w", encoding="utf-8") as json_file:
        json.dump([], json_file, ensure_ascii=False, indent=4)

# 初始化加速器
accelerator = Accelerator()

# 加載和處理 PDF 文檔
loader = PyPDFLoader(args.file_path)
docs = loader.load_and_split()
text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=128)
documents = text_splitter.split_documents(docs)

# 嵌入生成與向量存儲
embeddings = HuggingFaceEmbeddings(model_name=args.embedding_name)

# 檢查向量資料庫是否存在，決定載入或創建
if os.path.exists(args.vector_path):
    db = Chroma(persist_directory=args.vector_path, embedding_function=embeddings)
else:
    db = Chroma.from_documents(documents, embedding=embeddings, persist_directory=args.vector_path)

# 初始化檢索器
retriever = db.as_retriever(search_kwargs={"k": 1})

# 配置語言模型
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.bfloat16
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# 加載模型和分詞器
tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
config = AutoConfig.from_pretrained(args.model_name, hidden_activation="gelu_pytorch_tanh")
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    cache_dir=args.cache_dir,
    torch_dtype=torch_dtype,
    device_map="auto",
    quantization_config=quantization_config,
    trust_remote_code=True,
    config=config
)
model, tokenizer = accelerator.prepare(model, tokenizer)

# 配置 Hugging Face 管道
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    model_kwargs={"torch_dtype": torch_dtype},
)
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# 提示模板
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "你是一個高效且精準的助手，專門從參考資料中提取答案，解答使用者的問題。\n"
        "同時也是一個醫學專家，擁有非常多種疾病的成因，預防以及治療的觀念。\n"
        "請基於參考資料回答問題。如果參考資料中無法找到答案，請回應 \"無法從參考資料找到答案\"。\n"
        "參考資料：{context}\n\n使用者的問題：{question}"
    )
)

# 問答生成管道
llm_chain = prompt | llm | StrOutputParser()
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm_chain

# 問答處理與輸出
query = input("Enter questions about this file: ")
print("LLM generating answers...")
response = rag_chain.invoke(query)

# 格式化輸出
start_phrase = query
extracted_answer = response.split(start_phrase, 1)[-1].strip() if start_phrase in response else response
output_data = {"question": query, "answer": extracted_answer}

# 將生成的數據追加到 JSON 文件
with open(args.output_file_path, "r+", encoding="utf-8") as json_file:
    try:
        # 加載已有數據
        existing_data = json.load(json_file)
    except json.JSONDecodeError:
        existing_data = []  # 若文件為空，初始化為列表

    # 追加新數據
    existing_data.append(output_data)

    # 重寫文件內容
    json_file.seek(0)
    json.dump(existing_data, json_file, ensure_ascii=False, indent=4)
    json_file.truncate()

print("LLM ANSWER: ", response)
print(f"Save the generated answer in {args.output_file_path}")
