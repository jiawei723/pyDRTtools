# from pyDRTtools.GUI import launch_gui

# def main():
#     launch_gui()

# if __name__ == "__main__":
#     main()
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def launch_gui():
    st.title("pyDRTtools GUI")

    # 文件上传组件
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        # 读取CSV文件
        df = pd.read_csv(uploaded_file)
        st.write(df)

        # 绘制图表
        st.line_chart(df)

        # 其他处理逻辑
        if st.button("Process"):
            st.write("Processing...")
            # 添加数据处理和分析代码

def main():
    launch_gui()

if __name__ == "__main__":
    main()