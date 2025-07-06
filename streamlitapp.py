import streamlit as st
from page1 import page1
from page2 import page2
from page3 import page3

st.set_page_config(page_title="智能文本处理", layout="wide")

def main():
    # 初始化会话状态
    if "page1_output" not in st.session_state:
        st.session_state.page1_output = None
    if "page2_output" not in st.session_state:
        st.session_state.page2_output = None
    if "page3_output" not in st.session_state:
        st.session_state.page3_output = None

    st.sidebar.title("功能选择")
    page = st.sidebar.radio("选择功能", ["文本内容解析", "教学资源生成", "整合教学资源生成"])

    # 执行页面逻辑并获取输出
    if page == "文本内容解析":
        output = page1()
        if output:
            st.session_state.page1_output = output
    elif page == "教学资源生成":
        output = page2()
        if output:
            st.session_state.page2_output = output
    elif page == "整合教学资源生成":
        output = page3()
        if output:
            st.session_state.page3_output = output

    # 显示所有已生成的内容
    st.write("### 已生成的内容")
    
    if st.session_state.page1_output:
        st.subheader("文本内容解析 (Page 1)")
        st.markdown(st.session_state.page1_output)
    
    if st.session_state.page2_output:
        st.subheader("教学资源生成 (Page 2)")
        for model_name, content in st.session_state.page2_output:
            st.write(f"#### {model_name.capitalize()} 模型输出")
            st.markdown(content)
    
    if st.session_state.page3_output:
        st.subheader("整合教学资源生成 (Page 3)")
        st.markdown(st.session_state.page3_output)

if __name__ == "__main__":
    main()
