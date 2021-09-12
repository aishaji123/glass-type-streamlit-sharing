import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

feat_cols=['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
@st.cache
def prediction(model,feat_cols):
  glass_type=model.predict([feat_cols])
  glass_type=glass_type[0]
  if glass_type==1:
    return "building windows float processed".upper()
  elif glass_type==2:
    return "building windows non float processed".upper()
  elif glass_type==3:
    return "vehicle windows float processed".upper()
  elif glass_type==3:
    return "vehicle windows non float processed".upper()
  elif glass_type==4:
    return "vehicle windows float processed".upper()
  elif glass_type==5:
    return "containers".upper()
  elif glass_type==6:
    return "tableware".upper()
  else:
    return "headlamps".upper()

st.title("Glass Type Predictor")
st.sidebar.title("Glass Type Prediction Web App")

if st.sidebar.checkbox("Show raw data"):
  st.subheader("Glass Type Data set")
  st.dataframe(glass_df)

st.sidebar.subheader("Visualisation selector")
# Choosing x-axis values for the scatter plot.
# Add a multiselect in the sidebar with the 'Select the charts or plots:' label
# and pass all the 9 features as a tuple i.e. ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe') as options.
# Store the current value of this widget in the 'plot_list' variable.
plot_list=st.sidebar.multiselect("Select the charts or plots : ",("Correlation heatmap","Line plot","Area chart","Count plot","Pie chart","Box plot"))



#Displaying line plot
if "Line plot" in plot_list:
  st.subheader("Line plot")
  st.line_chart(glass_df)

#displaying area chart
if "Area chart" in plot_list:
  st.subheader("Area plot")
  st.area_chart(glass_df)

st.set_option('deprecation.showPyplotGlobalUse', False)
if "Correlation heatmap" in plot_list:
  st.subheader("Correlation heatmap")
  plt.figure(figsize=(10,5))
  sns.heatmap(glass_df.corr(),annot=True)
  st.pyplot()

if "Count plot" in plot_list:
  st.subheader("Count plot")
  sns.countplot(x="GlassType",data=glass_df)
  st.pyplot()

if "Pie chart" in plot_list:
  st.subheader("Pie chart")
  pie_data=glass_df["GlassType"].value_counts()
  plt.figure(figsize=(8,8))
  plt.pie(pie_data,labels=pie_data.index,autopct="%1.2f%%",startangle=30,explode=np.linspace(0.06,0.12,6))
  st.pyplot()

if "Box plot" in plot_list:
  st.subheader("Box plot")
  column=st.sidebar.selectbox("Select the columns for boxplot",('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType'))
  plt.figure(figsize=(12,3))
  sns.boxplot(glass_df[column])
  st.pyplot()