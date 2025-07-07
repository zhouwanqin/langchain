import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import pandas as pd
from io import BytesIO

# 设置 DashScope API 密钥
os.environ["DASHSCOPE_API_KEY"] = "sk-cb534124d0c44e7fb12dcb0271715482"

# 默认Prompts
CONTENT_PROMPT = """
    你是一个中医汉语教学专家。
    请你设计一个中医的情景文本，“{input}”文本，不要分点，在一个情景中用较为生活化的语言表达，多一些专业知识。
    以{context}为背景知识。
    示例如下：
        感冒
        小李这几天感觉身体不对劲，天气变化大，早上出门没注意添衣，晚上又淋了点雨，第二天就出现了嗓子干痒、鼻子里热热的、轻微头疼的症状。他量了体温，有点低烧，知道自己可能感冒了。
        他妈妈看决定带他去附近的中医诊所看看。老中医先是仔细询问了病情，包括症状出现的时间、性质以及是否有恶寒、发热等感受，并观察了他的面色和舌象。接着通过把脉，了解到小李的脉象浮数，这是典型的外感风热之象。根据中医理论，感冒主要是由于人体正气不足，卫外不固，导致风邪侵袭肺卫所致，而小李的情况属于风热犯肺。
        老中医为小李开具了一副中药方子，主要成分包括金银花（清热解毒）、连翘（疏散风热、清热解毒）、牛蒡子（疏散风热、利咽透疹）、薄荷（疏风散热）、桔梗（宣肺祛痰）等。这些药材组合起来，旨在辛凉解表，清热解毒，以达到驱散外邪、恢复肺卫功能的目的。
        同时，医生还建议小李多喝温开水，保持室内空气流通，避免再次受寒。此外，老中医提到可以适量食用一些具有润肺止咳作用的食物，如梨子，可以帮助缓解咳嗽和喉咙不适。回家后，小李妈妈按照医嘱给他熬制了药汤，并且煮了一碗姜糖水，帮助发汗解表。
        经过几天的调养，小李的症状逐渐减轻，体力也慢慢恢复。
"""
LANGUAGE_PROMPT = """
    你是一个中医汉语教学专家。
    汉语语言教学专家，擅长处理与“{input}”相关的汉语知识点问题，根据我给的这个内容进行语言点生成提取。
    以{context}为背景知识。
    要求：进行外国人学习汉语的语言点提取，配合简单的英文解读，要求有一般生词、专名术语、语法示例等，并且出几个简单的填空题
    示例如下：
        一般生词与短语
        添衣 (tiān yī) - Add clothes. To put on more clothes due to cold weather.
        淋了点雨 (lín le diǎn yǔ) - Got caught in a little rain. To get wet by rain slightly.
        嗓子干痒 (sǎng zi gàn yǎng) - Sore and itchy throat. A condition of the throat that feels dry and itchy.
        鼻子里热热的 (bí zi li rè rè de) - Nose feels hot. It refers to the sensation of having a warm or feverish nose.
        头疼 (tóu téng) - Headache. Pain in the head.
        把脉 (bǎ mài) - Take the pulse. Traditional Chinese medicine practice of feeling the pulse to diagnose illness.
        风热感冒 (fēng rè gǎn mào) - Wind-heat cold. A type of common cold characterized by symptoms like sore throat, nasal congestion with yellow mucus, etc.
        专名术语
        金银花 (jīn yín huā) - Honeysuckle. Known for its properties to clear heat and detoxify.
        连翘 (lián qiáo) - Forsythia suspensa. Used for clearing heat and resolving toxins.
        牛蒡子 (niú bàng zǐ) - Great burdock seed. Helps in relieving sore throat and promoting eruption of rashes.
        薄荷 (bò he) - Mint. Often used to disperse wind-heat and clear the head.
        桔梗 (jú gěng) - Balloon flower root. Commonly used for its effects on promoting lung function and expelling phlegm.
        迎香穴 (yíng xiāng xué) - Yingxiang acupoint. Located beside the nostrils, used for treating nasal congestion.
        风池穴 (fēng chí xué) - Fengchi acupoint. Found at the back of the neck, beneficial for headache and neck stiffness.
        语法示例
        得……一下
        例子：他妈妈看他蔫蔫的样子，就说：“这得赶紧调理一下，别拖成重感冒。”
        Translation: Seeing him looking listless, his mother said, "We need to treat this quickly before it turns into a severe cold."
        开了几副……主要是……
        例子：医生开了几副中药，主要是银花、连翘、牛蒡子这些清热解毒的药材。
        Translation: The doctor prescribed several doses of herbal medicine, mainly honeysuckle, forsythia, and great burdock seeds which are good for clearing heat and detoxifying.
        练习：
        他因为生病了，所以买了些中药___热解___。
        中医看病的时候会给病人___脉。
        他的鼻子里热热的，可能是___感冒。
        你最近睡觉前用___草泡脚吗？
        这个药方里有___翘和金___花。
    练习要求既包含一般生词与短语，又包含专名术语和语法示例。
"""
COGNITIVE_PROMPT = """
你是中医汉语教学专家，以{context}为背景知识。根据我提供的语言点内容，对{input}进行认知分析
示例如下：
题目内容：
阅读下列病例描述并完成任务：
“李某，女，35岁，最近经常失眠、多梦，情绪不稳定，容易生气，嘴里发干。中医认为这是‘阴虚火旺’引起的。”
1.找出三个表示身体不适的词组，并仿写一个你熟悉的症状表达。
2.用“是因为……引起的”结构改写这句话：“她经常失眠是因为肝火太旺。”
3.假设你是中医师，请用简单汉语给这位病人写一句建议：“如果你每天喝一些菊花茶，可能会……”
并设置一到两个小组交流活动：
1. 小组讨论：根据病例描述，讨论“阴虚火旺”的概念，并分享各自的理解。
2. 角色扮演：模拟中医问诊场景，学生分组扮演医生和病人，进行简单的问诊对话练习。
要求尽可能在形式上模仿我提供的内容。
"""
CULTURE_PROMPT = """
你是中医汉语教学专家，以{context}为背景知识，进行中医文化教学测试生成。
对感冒而言，示例如下，并提供参考答案：
1.在文化中医文化中，哪种理念强调预防疾病而不是治疗？
2.中医和西医对待感冒主要的治疗方法有何不同？
3.和你的文化相比，中医在感冒治疗上有哪些独特之处？
4.你觉得这种治疗方法在现代社会中是否仍然适用？为什么？
5.放眼全球，中医文化在感冒治疗方面有哪些影响力？请举例说明。
以上内容只是示例，你需要对我提供的背景知识进行处理！
"""

# 初始化语言模型
def create_llm(temperature=0.7, top_k=50):
    """初始化ChatOpenAI模型"""
    return ChatOpenAI(
        model="qwen-max-2025-01-25",
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

# 处理用户输入并生成教学资源
def process_input(user_input, uploaded_files, selected_models, temperature, top_k, rag_weight, progress_bar):
    """处理用户输入，运行选定模型并返回结果"""
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
    
    # 定义Prompts
    prompts = {
        "内容": PromptTemplate(input_variables=["context", "input"], template=CONTENT_PROMPT),
        "语言": PromptTemplate(input_variables=["context", "input"], template=LANGUAGE_PROMPT),
        "认知": PromptTemplate(input_variables=["context", "input"], template=COGNITIVE_PROMPT),
        "文化": PromptTemplate(input_variables=["context", "input"], template=CULTURE_PROMPT)
    }
    
    # 运行选中的模型
    results = []
    total_models = sum(1 for _, selected in selected_models if selected)
    current_step = 0
    for model_name, selected in selected_models:
        if selected:
            progress_bar.progress(0.95 + (current_step + 1) * 0.04 / max(total_models, 1), f"正在运行 {model_name} 模型...")
            prompt = prompts[model_name]
            final_input = prompt.format(context=context, input=user_input)
            response = llm.invoke(final_input)
            results.append((model_name, response.content))
            current_step += 1
    
    # 混合RAG和生成内容的权重
    if context and rag_weight > 0:
        results.insert(0, ("RAG参考内容", f"RAG参考内容 (权重 {rag_weight}):\n{context}"))
    
    progress_bar.progress(1.0, "完成！")
    return results

def page2():
    """教学资源生成页面"""
    st.title("教学资源生成")
    
    # 侧边栏：模型选择和参数设置
    with st.sidebar:
        st.header("模型选择")
        content_selected = st.checkbox("内容模型")
        language_selected = st.checkbox("语言模型")
        cognitive_selected = st.checkbox("认知模型")
        culture_selected = st.checkbox("文化模型")
        
        st.header("模型参数设置")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        top_k = st.slider("Top K", 1, 100, 50, 1)
        rag_weight = st.slider("RAG内容权重", 0.0, 1.0, 0.5, 0.1)
    
    # 主页面：文件上传和用户输入
    uploaded_files = st.file_uploader(
        "上传参考文件 (PDF/TXT/DOC/XLSX)",
        accept_multiple_files=True,
        type=['pdf', 'txt', 'docx', 'xlsx']
    )
    
    user_input = st.text_area("请输入教学资源需求：", height=150)
    
    # 开始按钮
    start_button = st.button("开始")
    
    # 处理逻辑
    if start_button:
        selected_models = [
            ("内容", content_selected),
            ("语言", language_selected),
            ("认知", cognitive_selected),
            ("文化", culture_selected)
        ]
        if user_input and any(selected for _, selected in selected_models):
            # 显示进度条
            progress_bar = st.progress(0.0)
            with st.spinner("正在生成教学资源..."):
                results = process_input(
                    user_input, uploaded_files, selected_models,
                    temperature, top_k, rag_weight, progress_bar
                )
            
            # 显示结果
            st.write("### 生成的教学资源")
            for model_name, output in results:
                st.subheader(f"{model_name.capitalize()} 模型输出")
                st.write(output)
            return results
        elif not user_input:
            st.warning("请输入教学资源需求后再点击开始！")
        elif not any(selected for _, selected in selected_models):
            st.warning("请至少选择一个模型后再点击开始！")
    return None

if __name__ == "__main__":
    page2()
