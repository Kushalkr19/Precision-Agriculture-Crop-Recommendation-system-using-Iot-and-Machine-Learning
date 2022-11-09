"""# **Streamlit dashboard creation**"""
import pickle
with open('model_pickle','rb') as f:
   lr = pickle.load(f)
with open('clf_pickle','rb') as f:
   rf = pickle.load(f)


import streamlit as st

st.write("""#Crop Recommendatation system
Predict the crop """)

from PIL import Image

image = Image.open('image.jpg')
st.image(image,caption='ML',use_column_width='True')

st.subheader('Data Information')

st.dataframe(df)

st.write(df.describe())

def get_user_input():
  N = st.sidebar.slider('Nitrogen',0,100)
  P = st.sidebar.slider('Phosporous',0,100)
  K = st.sidebar.slider('Potassium',0,100)
  temperature = st.sidebar.slider('Temperature',0,100)
  humidity = st.sidebar.slider('Humidity',0,100)
  rainfall = st.sidebar.slider('Rainfall',0,150)
  ph = st.sidebar.slider('ph',0,14)

  user_Data = {'N':N,
               'P':P,
               'K':K,
                'temperature':temperature,
               'rainfall':rainfall,
               'humidity':humidity,
               'ph':ph
               }
  features = pd.DataFrame(user_Data,index=[0])
  return features

user_input = get_user_input()
st.subheader('User Input:')
st.write(user_input)

st.subheader('Model Test Accuracy Score:')

st.write(str(rf.score(x_test, y_test)))

prediction1 = rf.predict(user_input)

st.subheader('Classification: ')
st.write(prediction1)

# Commented out IPython magic to ensure Python compatibility.
# %%writefile IOT.pynb
# st.write("""#Crop Recommendatation system
# Predict the crop """)
