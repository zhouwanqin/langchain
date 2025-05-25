import streamlit as st
import os
import pandas as pd
from io import BytesIO
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# 设置 DashScope API 密钥
os.environ["DASHSCOPE_API_KEY"] = "sk-cb534124d0c44e7fb12dcb0271715482"

# 默认Prompt
DEFAULT_PROMPT = """
你是教学框架生成器，接下来根据我给的内容和背景知识生成教学框架。以 Markdown 表格格式输出，包含 "4Cs"（内容、沟通、认知、文化）和 "教学策略"，并确保内容适合 CLIL 理论分析。
背景知识：{context}
用户输入：{input}
具体内容需要你自己丰富并完善，将示例替换为输入内容相关的。

若输入中医感冒相关文本，输出示例：

### 4Cs 教学框架

| 类别 | 教学目标 | 教学内容 |
|------|----------|----------|
| 内容 | 了解中医感冒 | 学习中医感冒的症状、诊断和治疗方法 |
| 沟通 | 掌握基本沟通、专业词汇、常用句型 | 1. 基础知识点：中医感冒的概念、症状、治疗方法<br>2. 语言点：中医术语、常用表达<br>3. 辅助题目测试：选择题、填空题等<br>4. 感冒问诊中可能出现的实际情况 |
| 认知 | 培养理解与思考判断、比较分析能力 | 1. 阅读所给病例描述并完成任务<br>2. 找出表达“感冒”的内容，并仿写一个你熟悉的症状表达<br>3. 用“是因为……引起的”结构改写句子<br>4. 假设你是中医师，提出简单建议 |
| 文化 | 理解中医与中华文化 | 唐朝中医感冒相关的历史小故事，体现“治未病”理念 |

### 教学策略

| 策略 |
|------|
| 1. 课文展示 |
| 2. 基础知识识记 |
| 3. 专门词汇学习 |
| 4. 基本句型学习 |
| 5. 认知提升练习 |
| 6. 模拟问诊提升 |
| 7. 文化拓展讨论 |

请以 Markdown 表格格式返回完整的教学框架内容。
"""
# 初始化语言模型
def create_llm(temperature=0.7, top_k=50):
    """初始化ChatOpenAI模型"""
    return ChatOpenAI(
        model="qwen-plus",
        openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        max_tokens=8192,
        temperature=temperature,
        extra_body={"top_k": top_k, "enable_thinking": False}
    )

# 加载和处理文件
def load_documents(uploaded_files):
    """加载并处理上传的文件（PDF/TXT/DOC/XLSX）"""
    documents = []
    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        temp_file_path = f"temp.{file_extension}"
        
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
            
        if file_extension == 'pdf':
            loader = PyPDFLoader(temp_file_path)
            documents.extend(loader.load())
        elif file_extension == 'txt':
            loader = TextLoader(temp_file_path)
            documents.extend(loader.load())
        elif file_extension == 'docx':
            loader = Docx2txtLoader(temp_file_path)
            documents.extend(loader.load())
        elif file_extension == 'xlsx':
            df = pd.read_excel(temp_file_path)
            text = df.to_string()
            documents.append(Document(page_content=text))
            
        os.remove(temp_file_path)
    
    return documents

# 创建RAG数据库
def create_rag_database(documents, progress_bar):
    """创建RAG向量数据库"""
    if not documents:
        return None
    
    # 更新进度：文本分割
    progress_bar.progress(0.3, "正在分割文档...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    # 更新进度：生成嵌入
    progress_bar.progress(0.6, "正在生成嵌入...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 更新进度：构建向量数据库
    progress_bar.progress(0.9, "正在构建向量数据库...")
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    return vectorstore

# 处理用户输入并生成教学框架
def process_input(user_input, uploaded_files, temperature, top_k, rag_weight, progress_bar):
    """处理用户输入并生成教学框架"""
    # 初始化模型
    progress_bar.progress(0.1, "初始化语言模型...")
    llm = create_llm(temperature, top_k)
    
    # 处理RAG
    vectorstore = None
    if uploaded_files:
        documents = load_documents(uploaded_files)
        vectorstore = create_rag_database(documents, progress_bar)
    
    # 获取RAG内容
    context = ""
    if vectorstore:
        progress_bar.progress(0.95, "正在检索RAG内容...")
        relevant_docs = vectorstore.similarity_search(user_input, k=3)
        context = "\n".join([doc.page_content for doc in relevant_docs])
    
    # 构建提示
    prompt = PromptTemplate(
        input_variables=["context", "input"],
        template=DEFAULT_PROMPT
    )
    
    # 混合内容
    final_input = prompt.format(context=context, input=user_input)
    
    # 生成响应
    progress_bar.progress(0.98, "正在生成教学框架...")
    response = llm.invoke(final_input)
    
    progress_bar.progress(1.0, "完成！")
    return response.content

def page1():
    """文本内容解析页面"""
    st.title("文本内容解析")
    
    # 侧边栏：模型参数设置
    with st.sidebar:
        st.header("模型参数设置")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1, key="page1_temperature")
        top_k = st.slider("Top K", 1, 100, 50, 1, key="page1_top_k")
        rag_weight = st.slider("RAG内容权重", 0.0, 1.0, 0.5, 0.1, key="page1_rag_weight")
    
    # 主页面：文件上传和用户输入
    uploaded_files = st.file_uploader(
        "上传文件 (PDF/TXT/DOC/XLSX)",
        accept_multiple_files=True,
        type=['pdf', 'txt', 'docx', 'xlsx'],
        key="page1_uploader"
    )
    
    user_input = st.text_area("请输入要解析的文本：", height=150, key="page1_input")
    
    # 开始按钮
    start_button = st.button("开始", key="page1_button")
    
    # 处理逻辑
    if start_button and user_input:
        # 显示进度条
        progress_bar = st.progress(0.0)
        with st.spinner("正在处理..."):
            response = process_input(
                user_input, uploaded_files, temperature, top_k, rag_weight, progress_bar
            )
        
        # 显示结果
        st.write("### 解析结果")
        st.markdown(response, unsafe_allow_html=True)
    elif start_button and not user_input:
        st.warning("请输入文本内容后再点击开始！")

if __name__ == "__main__":
    page1()
