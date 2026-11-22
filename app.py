import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from sklearn.datasets import load_iris

st.set_page_config(page_title="Dashboard Phan tich Du lieu", layout="wide")
st.title("He thong Phan tich Du lieu & Sinh Bao cao Khoa hoc Tu dong")

uploaded_file = st.sidebar.file_uploader("Tai len dataset (.csv)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_iris(as_frame=True).frame
    st.sidebar.info("Dang dung du lieu mau: Iris Dataset (UCI)")

df_clean = df.dropna()
st.subheader("1. Tong quan & Mo ta du lieu")
st.dataframe(df_clean.head(10), use_container_width=True)

st.subheader("2. Chon bien & Truc quan hoa du lieu")
numeric_cols = df_clean.select_dtypes(include=["float64", "int64"]).columns.tolist()

if len(numeric_cols) >= 2:
    c1, c2 = st.columns(2)
    x_var = c1.selectbox("Chon bien truc X:", numeric_cols, index=0)
    y_var = c2.selectbox("Chon bien truc Y:", numeric_cols, index=1)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.scatterplot(data=df_clean, x=x_var, y=y_var, ax=ax, color="#1F618D", s=60)
    ax.set_title(f"Tuong quan giua {x_var} va {y_var}")
    st.pyplot(fig)
    fig.savefig("temp_chart.png", dpi=150)

    st.subheader("3. Xuat bao cao PDF")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 15)
    pdf.cell(0, 10, "BAO CAO PHAN TICH DU LIEU KHOA HOC", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 7, f"Du lieu gom {df_clean.shape[0]} dong va {df_clean.shape[1]} cot.")
    pdf.image("temp_chart.png", x=20, w=170)
    pdf_bytes = bytes(pdf.output())

    st.download_button(
        label="Tai bao cao PDF hoan chinh",
        data=pdf_bytes,
        file_name="Bao_cao_khoa_hoc_AI.pdf",
        mime="application/pdf"
    )
