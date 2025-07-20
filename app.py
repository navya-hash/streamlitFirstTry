import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# toggle_status = st.toggle("Enable dark mode", value=True)

st.title("Diabetes dataset visualizations")
# step1(load the dataset)

df=pd.read_csv("diabetes.csv")
X=df.iloc[:,:-1]
Y=df.iloc[:,-1]
st.dataframe(df)
st.code(df.head())
st.subheader("Shape")
st.code(df.shape)
st.divider()
st.subheader("Null counts and data types")
data_null,dtypes=st.columns(2)

with data_null:
    st.code(X.isna().sum())
with dtypes:
    st.code(X.dtypes)

st.divider()
# if st.button('Show visulizations'):
if st.button("Show Visualizations"):
  pair=sns.pairplot(df,hue='Outcome')
  st.pyplot(pair)
  st.markdown("<br>", unsafe_allow_html=True)

  fig,axes=plt.subplots(nrows=2,ncols=4)
  fig.set_figwidth(18)
  for i in range(2):
      for j in range(4):
          sns.boxplot(x=df.iloc[:, i * 4 + j], ax=axes[i][j])  
          axes[i][j].set_title(df.columns[i * 4 + j])
        
  st.pyplot(fig)
  st.markdown("<br>", unsafe_allow_html=True)


  fig,axes=plt.subplots(nrows=2,ncols=4)
  fig.set_figwidth(18)
  for i in range(2):
      for j in range(4):
          sns.histplot(x=df.iloc[:, i * 4 + j], kde=True,ax=axes[i][j])  
          axes[i][j].set_title(df.columns[i * 4 + j])
        
  st.pyplot(fig)
st.markdown("""Observations:
            
            1)apply Robust scaler for Insulin,DiabetesPedigreeFunction,Age,BMI due to presence of more no 
            of outliers     2)apply Min Max Scaler for Pregnancies,Blood Pressure,Skin thickness for non normal distribution
            3) apply standard scaler for Glucose for normal distribution """)


preprocessor = Pipeline(
    [
        (
            "scale",
            ColumnTransformer(
                [
                    ("standard", StandardScaler(), ["Glucose"]),
                    ("robust", RobustScaler(), ["Insulin","DiabetesPedigreeFunction","Age","BMI"]),
                    
                ],
                remainder=MinMaxScaler(),
            ),
        ),
    ])

st.subheader("Dataset after preprocessing data")
X_transformed = preprocessor.fit_transform(X)
X_transformed=pd.DataFrame(X_transformed,columns=X.columns)
st.dataframe(X_transformed)








