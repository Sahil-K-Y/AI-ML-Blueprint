import streamlit as st
st.title("MY first streamlit")
st.write("Hello i am web app")
st.header("ye header hai")
st.subheader("ye subheader hai")
st.text("simple text")
st.success("success messege ")
st.warning("warning")
st.error("error")
st.info("info")

import pandas as pd
import numpy as np
st.subheader("Dataframe example")
df=pd.DataFrame({
    'Name':['SAhil','Arun','Ankush'],
    'Marks':[90,80,89]
})
st.dataframe(df)

data=pd.DataFrame(
    np.random.randn(20,2),columns=['A','B'])
st.line_chart(data)

st.sidebar.title("Sidebar")
st.sidebar.write("ye sidebar me")
name=st.sidebar.text_input("Name")
st.write("sidebar me name ",name)
age=st.slider("Age select kro",1,100,5)
st.write("meri age",age)
# Selectbox
option = st.sidebar.selectbox(
    "Subject choose karo",
    ["Math", "Science", "English"]
)
st.write("Tune choose kiya:", option)

# Button
if st.button("Click Me!"):
    st.balloons()  # 🎈 surprise!

# Checkbox
agree = st.checkbox("Main ML seekh raha hoon")
if agree:
    st.success("Bahut accha! Keep going! 🔥")

city=st.selectbox("CIty choose kro",["Delhi","Mumbai","Pune"])
st.write("Apki city",city)
st.number_input("Area enter kro")

if st.button("Predict"):
    st.success("button press kiya gya")
