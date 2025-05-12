import streamlit as st
import asyncio
from model_test import classify_prompt_with_llm, run_parallel_models, summarize_outputs

def main():
    st.title("CLIL指导的专门用途中医汉语平台")
    st.markdown("""
        欢迎使用CLIL指导的中医汉语助教系统！请输入您的需要，系统将生成一份完整清晰的教学设计方案
        包括：基础知识点、语言点、辅助题目测试、文化小故事等
    """)

    # 用户输入
    user_prompt = st.text_area("请输入您的需求：", height=150, placeholder="例如：我是老师，如何分析中医的感冒")

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