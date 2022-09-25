import pandas as pd
import numpy as np
import plotly_express as px
from PIL import Image
import seaborn as sns
sns.set_theme()

#kebutuhan model
from sklearn.model_selection import KFold, cross_val_score #untuk bagi dataset 
from sklearn.svm import SVC #untuk sentimen
import pickle
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer  #sebagai TF IDF
from sklearn import metrics #metode untuk pembentukan matriks 1x1 1x2 1x3
from sklearn.metrics import accuracy_score #metod untuk pembentukan skor akurasi 
from sklearn.model_selection import KFold #metode untuk perhitungan kfold
##packages untuk multiclass

from sklearn.multiclass import OneVsRestClassifier #pake library ovr
##from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier #untuk multioputput

#Bagian Tampilan
st.sidebar.title('Sentiment Analysis')
list_data = st.sidebar.selectbox('Pilih Menu :',
('Home','Data','Model Sentimen','Model Aspek','Visualisasi'))
if list_data == 'Home' or list_data =='':

@st.cache(persist=True)
def load_data():
    data = pd.read_csv(DATA_URL)
    data['sentimen_created'] = pd.to_datetime(data['sentimen'])
    return data

    data = load_data()

@st.cache(persist=True)
def plot_sentiment(ulasan):
    df = data[data['ulasan']== Opini]
    count = df['ulasan'].value_counts()
    count = pd.DataFrame({'Sentiment':count.index, 'Opini':count.values.flatten()})
    return count


st.sidebar.subheader("Breakdown Aspect by sentiment")
choice = st.sidebar.multiselect('Pick Ualasan', ('SVM','One vs All','Sentimen','Histogram'))
if len(choice) > 0:
    st.subheader("Breakdown Aspect by sentiment")
    breakdown_type = st.sidebar.selectbox('Visualization type', ['Pie chart', 'Bar plot', ], key='3')
    fig_3 = make_subplots(rows=1, cols=len(choice), subplot_titles=choice)
    if breakdown_type == 'Bar plot':
        for i in range(1):
            for j in range(len(choice)):
                fig_3.add_trace(
                    go.Bar(x=plot_sentiment(choice[j]).Sentiment, y=plot_sentiment(choice[j]).Sentimen, showlegend=False),
                    row=i+1, col=j+1
                )
        fig_3.update_layout(height=600, width=800)
        st.plotly_chart(fig_3)
    else:
        fig_3 = make_subplots(rows=1, cols=len(choice), specs=[[{'type':'domain'}]*len(choice)], subplot_titles=choice)
        for i in range(1):
            for j in range(len(choice)):
                fig_3.add_trace(
                    go.Pie(labels=plot_sentiment(choice[j]).Sentiment, values=plot_sentiment(choice[j]).Sentimen, showlegend=True),
                    i+1, j+1
                )
        fig_3.update_layout(height=600, width=800)

        st.markdown("<h2 style='text-align: center; color: grey ;'>CNN Indonesia Application </h2>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
                image = Image.open(r"C:\Users\user\Downloads\cnn.png")
                st.image(image)

        with col2:
                image = Image.open(r"C:\Users\user\Downloads\logo cnn.png")
                st.image(image, caption ='copyright : Desi Ariyani')

        with col3:
                 image = Image.open(r"C:\Users\user\Downloads\logo_sentimen.png")
                 st.image(image)

                  if submit_button:
                    st.subheader("PRE PROCESSING")
                #2. Case
                    st.info("Case Fold :")
                    case = kalimat.lower()
                    st.write('Hasil case folding :', case)
                #3. Removing 
                    st.info("Removing :")
                    import string 
                    import re #menggunakan regular expression

                    # import word_tokenize & FreqDist from NLTK
                    from nltk.tokenize import word_tokenize 
                    from nltk.probability import FreqDist
                    # Tokenizing 
                    # hapus hastag
                    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
                    text = re.sub(r"\d+", "", text)
                    st.write('Hasil :',text)

                     stopword untuk mewakili kata kata penting
                    st.info("Stopword :")
                    from nltk.corpus import stopwords
                    list_stopwords = stopwords.words('indonesian')
                    # memasukkan kamus stopword secara manual
                    list_stopwords.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 
                       'kalo', 'amp', 
                       'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                       '&amp', 'yah' ,'ter','anjenk','isinyaaa'])
                       # convert list to dictionary
                    list_stopwords = set(list_stopwords)
                    def stopwords_removal(words):
                        return [word for word in words if word not in list_stopwords]
                    sword = stopwords_removal(tokenisasi)
                    st.write(sword)
            
                #7.Normalisasi
                    st.info("Normalisasi :")
                    for index, row in data_normalisasi.iterrows():
                         if row[0] not in normalizad_word_dict:
                                normalizad_word_dict[row[0]] = row[1] 
                                def normalized_term(document):
                                    return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in document]
                    normal = normalized_term(sword)
                    st.write(normal)
                    
                    #8. Stemming
                    st.info("Stemming :")
                    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
                    hapus_number = re.sub(r"\d+", "", kalimat) #jaga" kalo ada nomornya
                    #buat factory
                    Stemmer = factory.create_stemmer()
                    # stemmed
                    steming = Stemmer.stem(hapus_number)
                    st.write(steming)

                #8. Masukkan pembobotan
                    st.info("Term Frequency :")
                    import pickle
                   
                    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open(r"C:\Users\user\Documents\Skripsi File\Coding\TFIDF-sentimen.pkl", "rb")))
                    tfidf = transformer.fit_transform(loaded_vec.fit_transform([steming]))
                    st.write('Hasil Pembobotan : ',tfidf)

                    #9. Masukkan sentimen 
                    loaded_SVM_model = pickle.load(open(r"C:\Users\user\Documents\Skripsi File\Coding\SVMLinierSentimen_model.sav", 'rb'))
                    st.info("Keterangan Model Support Vector Machine : ")
                    st.write(loaded_SVM_model)
                    st.info("Hasil Sentimen..")
                    hasil = loaded_SVM_model.predict(tfidf)
                    st.write('Sentimen dari teks ulasan:', [hasil])