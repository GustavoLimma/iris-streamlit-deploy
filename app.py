# data app for iris prediction (classification)
import streamlit as st
# in case of problems with tensorflow try to add this env variable
#import os
#os.environ["KERAS_BACKEND"] = "tensorflow"

from keras.api.models import load_model

#from keras.models import load_model (old)
import numpy as np

from joblib import load

model_nn = load_model('models/model.keras')
model_knn = load('models/model-knn.pkl')

# assumindo este labelEncoder
class_labels = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']

# st.set_page_config(layout="wide")

st.title('Classificação de Flor :blue[Iris]')

petal_length_input = st.slider("Comprimento da pétala", 1.0, 6.9, (1.0+6.9)/2)
petal_width_input = st.slider("Espessura da pétala", 0.1, 2.5, (0.1+2.5)/2)

btn = st.button('Pergunte à IA')

modelo_escolhido = st.selectbox("Escolha o modelo de IA", ["Rede Neural", "KNN"])

st.markdown("""
<style>
.big-font {
    font-size:300px !important;
}
.answer-color {
    color:yellow;
}
</style>
""", unsafe_allow_html=True)

# st.markdown('<p class="big-font">Hello World !!</p>', unsafe_allow_html=True)

if btn:
    if modelo_escolhido == "Rede Neural":
        classes_prob = model_nn.predict(
            np.array([[petal_length_input, petal_width_input]]))
        if (np.argmax(classes_prob) == 0):
            st.image('img/setosa.jpg')
        elif (np.argmax(classes_prob) == 1):
            st.image('img/versicolor.jpg')
        else:
            st.image('img/virginica.jpg')
        text = class_labels[np.argmax(classes_prob)]
        st.html(
        f"""<p class="answer-color">{class_labels[np.argmax(classes_prob)]}</p>""")
    else:
    # O iris_class é um array, então não podemos comparar com "if (iris_class == 1):", ele não retornará verdadeiro ou falso.
        iris_class = model_knn.predict(
        np.array([[petal_length_input, petal_width_input]]))
    # Atribui o array a uma variável, mas poderia ser passado direto: if iris_class[0] == 0:
        predicted = iris_class[0]
    # Indice 0 porque iris_class é um vetor com uma só posição, a do resultado do modelo.
        if predicted == 0:
            st.image('img/setosa.jpg')
        elif predicted == 1:
            st.image('img/versicolor.jpg')
        else:
            st.image('img/virginica.jpg')

        st.markdown(
            f"""<p class="answer-color">{class_labels[predicted]}</p>""", unsafe_allow_html=True)
    # Não funcionou:
    #    iris_class = model_knn.predict(
    #        np.array([[petal_length_input, petal_width_input]]))
    #    if (iris_class == 1):
    #        st.image('img/versicolor.jpg')
    #    elif (iris_class == 0):
    #        st.image('img/setosa.jpg')
    #    else:
    #        st.image('img/virginica.jpg')
    #    text = class_labels[iris_class[0]]
    #    st.html(
    #    f"""<p class="answer-color">{class_labels[np.argmax(iris_class)]}</p>""")
    
st.caption('IA pode cometer erros. Considere verificar informações importantes')
# some_number = st.number_input('Enter a number')

