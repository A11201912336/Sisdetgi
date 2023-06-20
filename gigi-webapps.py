import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier

# tampilan web
st.write(""" 
# Classification of Dental Diseases (Web Apps)
Web-based application for predicting (classifying) Human Dental Disease 
""")

img = Image.open('gigie.png')
img = img.resize((700,418))
st.image(img, use_column_width=False)

# Upload File Excel untuk parameter inputan
st.sidebar.image('talk-to.jpg')
st.sidebar.header('Upload your Excel file')

upload_file = st.sidebar.file_uploader('')
if upload_file is not None:
    inputan = pd.read_excel(upload_file)
else:
    def input_user():
        st.sidebar.text('')
        st.sidebar.header('INPUTAN USER')
        st.sidebar.image('kett.png')
        
        col1, col2 = st.sidebar.columns(2)
        with col1:           
            gender = st.selectbox('Gender', ('0', '1'))
            st.caption('0 = Laki-Laki 1 = Perempuan')

        with col2:
            umur = st.number_input('Umur', value=0)
            st.caption('usia anda')

        col3, col4, col5 = st.sidebar.columns(3)
        with col3:        
            demam = st.number_input('demam', min_value=0.0, max_value=1.0)
        with col4:
            pusing = st.number_input('pusing',min_value=0.0, max_value=1.0)
        with col5:
            gusi_bengkak = st.number_input('gusi bengkak',min_value=0.0, max_value=1.0)
        with col3:
            nyeri_telinga = st.number_input('nyeri telinga',min_value=0.0, max_value=1.0)
        with col4:
            gigi_nyeri = st.number_input('gigi nyeri',min_value=0.0, max_value=1.0)
        with col5:
            sulit_mengunyah = st.number_input('sulit mengunyah',min_value=0.0, max_value=1.0)
        with col3:
            gusi_bernanah = st.number_input('gusi bernanah',min_value=0.0, max_value=1.0)
        with col4:
            gusi_berdarah = st.number_input('gusi berdarah',min_value=0.0, max_value=1.0)
        with col5:
            pipi_bengkak = st.number_input('pipi bengkak',min_value=0.0, max_value=1.0)
        with col3:
            tenggorokan_bengkak = st.number_input('tenggorokan bengkak',min_value=0.0, max_value=1.0)
        with col4:
            bau_nafas = st.number_input('bau nafas',min_value=0.0, max_value=1.0)
        with col5:
            gigi_berkerak = st.number_input('gigi berkerak',min_value=0.0, max_value=1.0)
        with col3:
            gigi_berubah_warna = st.number_input('gigi berubah warna',min_value=0.0, max_value=1.0)
        with col4:
            gusi_perih = st.number_input('gusi perih',min_value=0.0, max_value=1.0)
        with col5:
            gusi_menyusut = st.number_input('gusi menyusut',min_value=0.0, max_value=1.0)
        with col3:
            gigi_berlubang = st.number_input('gigi berlubang',min_value=0.0, max_value=1.0)
        with col4:
            gusi_merah = st.number_input('gusi merah',min_value=0.0, max_value=1.0)
        with col5:
            muncul_plak_keras = st.number_input('muncul plak keras',min_value=0.0, max_value=1.0)
        with col3:
            sulit_menelan = st.number_input('sulit menelan',min_value=0.0, max_value=1.0)
        with col4:
            pembusukan_gigi = st.number_input('pembusukan gigi',min_value=0.0, max_value=1.0)
        with col5:
            gigi_sensitif = st.number_input('gigi sensitif',min_value=0.0, max_value=1.0)
        with col3:
            gigi_goyang = st.number_input('gigi goyang',min_value=0.0, max_value=1.0)
        with col4:
            air_liur_pahit = st.number_input('air liur pahit',min_value=0.0, max_value=1.0)
        with col3:
            noda_hitam_di_gigi = st.number_input('noda hitam di gigi',min_value=0.0, max_value=1.0)
        with col4:
            gigi_bertumpuk = st.number_input('gigi bertumpuk',min_value=0.0, max_value=1.0)

        data = {'umur' : umur,
                'gender' : gender,
                'demam' : demam,
                'pusing' : pusing,
                'gusi_bengkak' : gusi_bengkak,
                'nyeri_telinga' : nyeri_telinga,
                'gigi_nyeri' : gigi_nyeri,
                'sulit_mengunyah' : sulit_mengunyah,
                'gusi_bernanah' : gusi_bernanah,
                'gusi_berdarah' : gusi_berdarah,
                'pipi_bengkak' : pipi_bengkak,
                'tenggorokan_bengkak' :tenggorokan_bengkak,
                'bau_nafas' :bau_nafas,
                'gigi_berkerak' :gigi_berkerak,
                'gigi_berubah_warna' :gigi_berubah_warna,
                'gusi_perih' :gusi_perih,
                'gusi_menyusut' :gusi_menyusut,
                'gigi_berlubang' :gigi_berlubang,
                'gusi_merah' : gusi_merah,
                'muncul_plak_keras' : muncul_plak_keras,
                'sulit_menelan' : sulit_menelan,
                'pembusukan_gigi' : pembusukan_gigi,
                'gigi_sensitif' : gigi_sensitif,
                'gigi_goyang' : gigi_goyang,
                'air_liur_pahit' : air_liur_pahit, 
                'noda_hitam_di_gigi' : noda_hitam_di_gigi,
                'gigi_bertumpuk' : gigi_bertumpuk}
        fitur = pd.DataFrame(data, index=[0])
        return fitur
    inputan = input_user()

# Menggabungkan input dan dataset
dents_raw = pd.read_excel("newdatagigi.xlsx")
dents_prediction = dents_raw.drop(columns=['diagnosa'])
df = pd.concat([inputan, dents_prediction], axis=0)
df = df[:1] #ambil data baris pertama

# Menampilkan parameter hasil inputan
st.subheader('Parameter of Input')

if upload_file is not None:
    st.write(df)
else:
    st.write('Waiting for the excel file to upload..')
    st.write(df)

# Load save model
load_model = pickle.load(open('DeteksiGigi_Model.pkl', 'rb'))

# Terapkan Random Forest
prediction = load_model.predict(df)

st.subheader('Class Label Description')
detections = np.array(['Abses Gigi','Karang Gigi', 'Karies Gigi','Periodontitis', 'Impaksi Gigi','Gingivitis', 'Pulpitis', 'Erosi Gigi'])
st.write(detections)

st.subheader('''Prediction Results (Classification)''')
st.subheader('''Human Dental Diseases''')
st.write(detections[prediction])

st.subheader('''Tabel Deskripsi Penyakit dan Pengobatannya''')
dents_desc= pd.read_excel("penjelasan.xlsx")
