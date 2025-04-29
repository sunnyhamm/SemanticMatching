import os
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide", page_title="Variable Matching Module")
st.header("Census Variable Matching App")


main_row1 = st.columns(2, vertical_alignment="top")
with main_row1[0].container(border=True, height=1000):

   with st.form("variable_matching_form", border=False):

        row1 = st.columns(1, vertical_alignment="top")
        with row1[0].container(border=False):
            st.write("Select Input File :")

        row2 = st.columns(1, vertical_alignment="top")
        with row2[0].container():
            uploaded_file = st.file_uploader("input file", type=['csv', 'txt'],accept_multiple_files=False, label_visibility="hidden")
            if uploaded_file is not None:
                file_name = uploaded_file.name

        row3 = st.columns(1, vertical_alignment="top")
        with row3[0].container():
            dedupe_columns = st.multiselect(
                                        "Remove Duplicates :",
                                        ["ID", "Label", "Description", "Comment"],
                                        [],
                                        )

        row4 = st.columns(2, vertical_alignment="top")
        with row4[0].container():
            st.write("Remove NULLS / Literals :")

        with row4[1].container(border=True):
            remove_values = st.text_input("label", label_visibility="hidden")

        row5 = st.columns(2, vertical_alignment="top")
        with row5[0].container():
            st.write("In Column :")

        with row5[1].container():
            remove_values_column = st.selectbox("rem values",("ID", "Label", "Description", "Comment"),label_visibility="hidden")


        submitted = st.form_submit_button("Match")
        # if submitted:
        #     st.write("Submitted")

        row6 = st.columns(1, vertical_alignment="top")
        with row6[0].container():
            st.write("Input Data")
            df1 = pd.DataFrame(np.random.randn(50, 20), columns=("col %d" % i for i in range(20)))
            st.dataframe(df1)




with main_row1[1].container(border=True, height=1000):

    st.write("Matching Results")
    df2 = pd.DataFrame(np.random.randn(50, 20), columns=("col %d" % i for i in range(20)))
    st.dataframe(df2)





