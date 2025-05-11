import os
import asyncio
import streamlit as st
import json
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate

# 设置 DashScope API 密钥（建议使用环境变量）
os.environ["DASHSCOPE_API_KEY"] = "sk-cb534124d0c44e7fb12dcb0271715482"

# 初始化模型
def create_llm():
    return ChatOpenAI(
        model="qwen-plus",
        openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        max_tokens=1024,
        extra_body={"enable_thinking": False}
    )

# 使用大模型进行 Prompt 分类
def classify_prompt_with_llm(prompt):
    llm = create_llm()
    categories = ["中医基础知识", "语言", "中医文化", "中医认知"]
    
    classification_prompt = PromptTemplate.from_template(
        """
        你是一位智能分类器，擅长分析用户输入的问题并判断其所属类别。现有以下类别：
        - 中医基础知识：涉及针灸、中药、经络、穴位、方剂、诊断等实用知识。
        - 语言：涉及翻译、对话、表达优化、语法等语言相关问题。
        - 中医文化：涉及中医历史、哲学、阴阳、五行等文化和思想内涵。
        - 中医认知：涉及中医的思维方式、理论框架、观念等抽象认知。
        
        用户输入的问题是：“{prompt}”
        
        请分析该问题，判断它属于哪些类别（可能属于多个类别）。返回一个 JSON 格式的列表，包含匹配的类别名称。如果没有明确匹配，默认返回 ["中医认知"]。
        
        示例：
        输入：“请你介绍中医的感冒？”
        输出：```json
        ["中医文化", "中医认知","中医认知"]
        ```
        输入：“如何用中药治疗感冒？”
        输出：```json
        ["中医基础知识"]
        ```
        """
    )
    
    messages = [SystemMessage(content=classification_prompt.format(prompt=prompt))]
    try:
        response = llm.invoke(messages)
        # 尝试解析 JSON 输出
        matched_categories = json.loads(response.content)
        # 验证类别有效性
        matched_categories = [cat for cat in matched_categories if cat in categories]
        return matched_categories if matched_categories else ["中医认知"]
    except (json.JSONDecodeError, ValueError):
        # 如果 JSON 解析失败或输出无效，默认返回 ["中医认知"]
        st.warning("分类器输出格式错误，默认归类为‘中医认知’。")
        return ["中医认知"]

# 定制每个类别的 Prompt
def get_category_prompt(category, user_prompt):
    prompts = {
        "中医基础知识":
             SystemMessage(
                 content="""
                 你是一位中医专家，擅长解释针灸、中药、经络等实用知识。请以专业、准确的语言回答，注重实用性和科学依据：
                 """
                 ),
        "语言": 
            SystemMessage(
                content="""
                你是一位语言专家，擅长处理翻译、表达优化等语言问题。请以清晰、简洁的语言回答，注重表达的流畅性：
                """
                ),
        "中医文化": 
            SystemMessage(
                content="""
                你是一位中医文化研究者，熟悉中医历史、哲学和阴阳五行等内涵。请以深入且引人入胜的方式回答，突出文化价值：
                """
                ),
        "中医认知": 
            SystemMessage(
                content="""
                你是一位中医理论家，擅长分析中医的思维方式和理论框架。请以逻辑清晰、富有洞察力的方式回答，强调理论深度：
                """
                )
    }
    return [prompts[category], HumanMessage(content=user_prompt)]

# 异步运行单个模型
async def run_model(llm, messages):
    try:
        response = await llm.ainvoke(messages)
        return response.content
    except Exception as e:
        st.error(f"模型运行错误：{str(e)}")
        return "模型运行失败，请检查 API 配置或网络连接。"

# 并行运行所有模型
async def run_parallel_models(prompt, categories):
    llm = create_llm()
    tasks = []
    for category in categories:
        messages = get_category_prompt(category, prompt)
        tasks.append(run_model(llm, messages))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return dict(zip(categories, results))

# 总模型提炼输出
def summarize_outputs(prompt, category_outputs):
    llm = create_llm()
    summary_prompt = PromptTemplate.from_template(
        """
        你是一位综合分析专家，擅长从多角度提炼信息。以下是针对用户问题“{prompt}”的分类回答：
        {category_outputs}
        
        请综合以上信息，生成一个简洁、连贯且全面的回答，突出核心观点，避免冗余。回答应：
        - 结构清晰，分点或段落组织。
        - 语言流畅，适合普通用户理解。
        - 保留各分类视角的独特贡献。
        """
    )
    messages = [
        SystemMessage(content=summary_prompt.format(
            prompt=prompt,
            category_outputs="\n".join([f"{cat}: {out}" for cat, out in category_outputs.items()])
        ))
    ]
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        st.error(f"提炼模型运行错误：{str(e)}")
        return "提炼失败，请检查 API 配置或网络连接。"

# Streamlit 界面
def main():
    st.title("中医问题智能分析系统")
    st.markdown("""
        欢迎使用中医问题分析系统！请输入您的问题，系统将：
        1. 使用大模型分析问题所属类别（中医基础知识、语言、中医文化、中医认知）。
        2. 从对应视角生成专业回答。
        3. 综合提炼出简洁、全面的最终回答。
    """)

    # 用户输入
    user_prompt = st.text_area("请输入您的问题：", height=150, placeholder="例如：中医的阴阳理论如何影响诊断？")

    if st.button("提交", key="submit"):
        if not user_prompt:
            st.error("请输入问题！")
            return

        # 显示用户输入
        st.subheader("您的问题：")
        st.write(user_prompt)

        # 使用大模型分类
        with st.spinner("正在分析问题类别..."):
            categories = classify_prompt_with_llm(user_prompt)
        st.subheader("问题分类：")
        st.write(", ".join(categories))

        # 并行运行模型
        with st.spinner("正在生成分类回答..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            category_outputs = loop.run_until_complete(run_parallel_models(user_prompt, categories))
            loop.close()

        # 显示各模型输出（调试用，可注释）
        st.subheader("各模型输出（调试用）：")
        for category, output in category_outputs.items():
            with st.expander(f"{category}"):
                st.write(output)

        # 提炼最终回答
        with st.spinner("正在提炼最终回答..."):
            final_answer = summarize_outputs(user_prompt, category_outputs)
        
        # 显示最终回答
        st.subheader("最终回答：")
        st.markdown(final_answer)

if __name__ == "__main__":
    main()