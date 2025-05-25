import streamlit as st
import asyncio
from model_test import analyze_prompt, run_parallel_models, summarize_outputs

def main():
    st.title("CLIL指导的专门用途中医汉语平台")
    st.markdown("""
        欢迎使用CLIL指导的中医汉语助教系统！请输入您的需求，系统将生成一份完整清晰的教学设计方案，
        包括：基础知识点、语言点、辅助题目测试、文化小故事等。
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

        # 使用大模型分析输入，提取知识点和激活类别
        with st.spinner("正在分析问题..."):
            knowledge_point, activated_categories = analyze_prompt(user_prompt)
        st.subheader("提取信息：")
        st.write(f"知识点：{knowledge_point}")
        st.write("激活类别：" + ", ".join(activated_categories))

        # 并行运行模型，获取各类别输出
        with st.spinner("正在生成各分类回答..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            category_outputs = loop.run_until_complete(run_parallel_models(knowledge_point, activated_categories))
            loop.close()

        # 显示各模型输出（调试用，可选择隐藏）
        st.subheader("各模型输出（调试用）：")
        for category, output in category_outputs.items():
            with st.expander(f"{category}"):
                st.write(output)

        # 提炼最终回答
        with st.spinner("正在提炼最终回答..."):
            final_answer = summarize_outputs(user_prompt, category_outputs, activated_categories)

        # 显示最终回答
        st.subheader("最终回答：")
        st.markdown(final_answer)

if __name__ == "__main__":
    main()
