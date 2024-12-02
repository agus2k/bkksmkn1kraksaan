import streamlit as st

def prediksi():
    import streamlit as st
    import pandas as pd
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import sklearn
    import time
    import seaborn as sn


    st.title('Prediksi Gaji pada saat lulus')
    st.subheader('Memprediksi apakah kalian lulus lalu bekerja dengan gaji di atas atau sama dengan UMR tempat kalian bekerja')
    url = 'https://raw.githubusercontent.com/agus2k/gcc_data_menntah/main/Data_Mentah_Bekerja.csv'

    df_kerja = pd.read_csv(url,
            delimiter=';', 
            header='infer', 
            index_col=False)
    X = df_kerja[['Jenis_Kelamin','Jurusan','Kota','Linier','Kerjasama','BKK']]
    Y = df_kerja['UMR']
    from sklearn.model_selection import train_test_split

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,random_state=123) # 75% training and 25% test
    #Import svm model
    from sklearn import svm
    #Create a svm Classifier
    clf_svm = svm.SVC(kernel='linear',probability=True) # Linear Kernel
    #Train the model using the training sets
    clf_svm.fit(X_train, y_train.values.ravel())
    #Predict the response for test dataset

    from sklearn.naive_bayes import GaussianNB
    # Mengaktifkan/memanggil/membuat fungsi klasifikasi Naive Bayes
    modelnb = GaussianNB()
    # Memasukkan data training pada fungsi klasifikasi Naive Bayes
    clf_nb = modelnb.fit(X_train, y_train.values.ravel())    

    from sklearn.neighbors import KNeighborsClassifier  
    clf_knn = KNeighborsClassifier(n_neighbors=5)  
    clf_knn.fit(X_train, y_train.values.ravel())

    nama = st.text_input('Nama')
    jk = st.radio('Jenis Kelamin',options=('Laki-Laki','Perempuan'))
    jurusan = st.selectbox('Jurusan', options=['Akuntansi dan Keuangan Lembaga','Bisnis Daring dan Pemasaran','Otomatisasi dan Tata Kelola Perkantoran','Multimedia','Rekayasa Perangkat Lunak','Teknik Komputer dan Jaringan'])

    kota = st.selectbox('Kota tempat kalian akan bekerja', options=['Probolinggo','Kota Probolinggo','Jember','Gresik'])

    linier = st.selectbox('Pekerjaan Linier dengan jurusan', options=['Ya','Tidak'])

    kerjasama = st.selectbox('Tempat bekerja yang kalian inginkan bekerja sama dengan sekolah', options=['Ya','Tidak'])

    bkk = st.selectbox('Butuh bantuan BKK untuk masuk ke tempat kerja ?', options=['Ya','Tidak'])

    if st.button('Prediksi'):
        mybar = st.progress(0)
        jk = 0 if jk == 'Laki-Laki' else 1
        if jurusan == 'Akuntansi dan Keuangan Lembaga':
            jurusan = 1    
        elif jurusan == 'Bisnis Daring dan Pemasaran':
            jurusan = 2
        elif jurusan == 'Otomatisasi dan Tata Kelola Perkantoran':
            jurusan = 3
        elif jurusan == 'Multimedia':
            jurusan = 4
        elif jurusan == 'Rekayasa Perangkat Lunak':
            jurusan = 5
        elif jurusan == 'Teknik Komputer dan Jaringan':
            jurusan = 6
        kotaa = 0
        if kota == 'Probolinggo':
            kotaa =1
        elif kota == 'Kota Probolinggo':
            kotaa = 2
        elif kota == 'Jember':
            kotaa = 3
        elif kota == 'Gresik':
            kotaa = 4
        
        linier = 1 if linier == 'Ya' else 0
        
        kerjasama = 1 if kerjasama == 'Ya' else 0

        bkk = 1 if bkk == 'Ya' else 0
        tes = [[jk,jurusan,kotaa,linier,kerjasama,bkk]]

        
        hasil_svm = clf_svm.predict(tes)
        akurasi_svm = clf_svm.predict_proba(tes)
        hasil_nb = clf_nb.predict(tes)
        akurasi_nb = clf_nb.predict_proba(tes)
        hasil_knn = clf_knn.predict(tes)
        akurasi_knn = clf_knn.predict_proba(tes)

        for persen in range(100):
            time.sleep(0.01)
            mybar.progress(persen+1)

        st.subheader('Support Vector Machine')
        if hasil_svm[0]==1:
            st.write('Dengan menggunakan metode SVM maka {} akan bekerja dengan **GAJI DIATAS/SAMA DENGAN UMR** wilayah {} dengan tingkat peluang {}%'.format(nama,kota,round(akurasi_svm[0][1]*100),3))
        else:
            st.write('Dengan menggunakan metode SVM maka {} akan bekerja dengan **GAJI DIBAWAH UMR** wilayah {} dengan tingkat peluang {}%'.format(nama,kota,round(akurasi_svm[0][1]*100),3))

        st.subheader('Naive Bayes')
        if hasil_nb[0]==1:
            st.write('Dengan menggunakan metode Naive Bayes maka {} akan bekerja dengan **GAJI DIATAS/SAMA DENGAN UMR** wilayah {} dengan tingkat peluang {}%'.format(nama,kota,round(akurasi_nb[0][1]*100),3))
        else:
            st.write('Dengan menggunakan metode Naive Bayes maka {} akan bekerja dengan **GAJI DIBAWAH UMR** wilayah {} dengan tingkat peluang {}%'.format(nama,kota,round(akurasi_nb[0][1]*100),3))

        st.subheader('K-Nearest Neighbor')
        if hasil_knn[0]==1:
            st.write('Dengan menggunakan metode KNN maka {} akan bekerja dengan **GAJI DIATAS/SAMA DENGAN UMR** wilayah {} dengan tingkat peluang {}%'.format(nama,kota,round(akurasi_knn[0][1]*100),3))
        else:
            st.write('Dengan menggunakan metode KNN maka {} akan bekerja dengan **GAJI DIBAWAH UMR** wilayah {} dengan tingkat peluang {}%'.format(nama,kota,round(akurasi_knn[0][1]*100),3))
        
def infografis():
    import streamlit as st
    import pandas as pd
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import sklearn
    import time
    import seaborn as sn

    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title("Penelusuran Tamatan SMKN 1 Kraksaan tahun 2021") #menampilkan halaman utama
    st.header('Dengan menggunakan Hasil Penelusuran tamatan, maka saya akan memvisualisasikan data hasil penelusuran tamatan')
    url = 'https://raw.githubusercontent.com/agus2k/gcc_data_menntah/2a71f4a4c0460604e0a546dc0221ad11da379f9b/Data_Mentah_Keseluruhan.csv'
    df = pd.read_csv(url,
        delimiter=';', 
        header='infer', 
        index_col=False)
    st.table(df.head(5))
    counts = [
            df[df['Jenis_Kelamin']==0].shape[0],
            df[df['Jenis_Kelamin']==1].shape[0]
    ]
    #print(counts)
    labels = ['Laki - Laki','Perempuan']
    fig, ax = plt.subplots()
    ax.pie(counts, labels=labels, autopct='%1.1f%%')
    ax.axis('equal')
    st.subheader('Jenis Kelamin Alumni 2021 yang mengisi survei')

    st.pyplot(fig)

    plt.figure(figsize=(5,7))
    plt.bar(labels, counts, color=['blue','red'])

    plt.title('Jumlah Alumni Yang Mengisi Survei Berdasarkan Gender', size=10)
    plt.ylabel('Jumlah Alumni', size=9)
    plt.xlabel('Gender', size=12)
    plt.xticks(size=9)
    plt.yticks(size=12)

    st.pyplot()
    st.write('Dari Sini kita bisa melihat bahwa yang mengisi survei lebih banyak dari Jenis Kelamin Perempuan dibanding Laki - Laki. Disini bisa dimaklumi dikarenakan memang siswa SMKN 1 Kraksaan mayoritas gender lebih banyak Perempuan dibanding Laki-Laki')
    st.header('Sebaran Alumni')
    st.subheader('Lalu kita buat Pie Chart dan Bar Chart Berdasarkan Pekerjaan mereka pada saat lulus, disini ada 5 pilihan yaitu :')
    st.write('1. Kuliah/Kursus (Melanjutkan Studi)')
    st.write('2. PNS/ASN')
    st.write('3. Karyawan Swasta')
    st.write('4. Wirausaha')
    st.write('5. Belum Bekerja')
    counts = [
            df[df['Pekerjaan']==1].shape[0],
            df[df['Pekerjaan']==2].shape[0],
            df[df['Pekerjaan']==3].shape[0],
            df[df['Pekerjaan']==4].shape[0],
            df[df['Pekerjaan']==5].shape[0],
    ]
    labels = ['Kuliah/Kursus','PNS','Karyawan Swasta','Wirausaha','Belum Bekerja']
    fig, ax = plt.subplots()
    ax.pie(counts, labels=labels, autopct='%1.1f%%')
    ax.axis('equal')
    plt.title('Pekerjaan Alumni tahun 2021')

    st.pyplot(fig)

    plt.figure(figsize=(15,7))
    plt.bar(labels, counts, color=['blue','red','purple','green','navy'])

    plt.title('Pekerjaan Alumni 2021', size=16)
    plt.ylabel('Jumlah Alumni', size=14)
    plt.xlabel('Pekerjaan', size=12)
    plt.xticks(size=9)
    plt.yticks(size=12)

    st.pyplot()
    st.write('Disini kita bisa simpulkan bahwa alumni yang lulus tahun 2021 paling banyak Belum bekerja, disusul Kuliah/Kursus, lalu Karyawan, Wirausaha dan paling sedikit adalah PNS/ASN')
    st.write('Belum Bekerja memang paling banyak dikarenakan pada saat mereka lulus kondisi masih pandemi serta perusahaan yang biasa membuka lowongan pun tidak membuka lowongan dikarenakan pandemi Covid 19')
    pekerjaan = []
    for i in range(1, 6):
        d = []
        for j in range(1,7):
            d.append(df[(df.Jurusan == j) & (df.Pekerjaan == i)].count()['Pekerjaan'])
        pekerjaan.append(d)

    jurusan = ['Akuntansi dan Keuangan Lembaga',
            'Bisnis Daring dan Pemasaran',
            'Otomatisasi dan Tata Kelola Perkantoran',
            'Multimedia',
            'Rekayasa Perangkat Lunak',
            'Teknik Komputer dan Jaringan']
    x = np.arange(len(jurusan))
    width = 0.15

    fig, ax = plt.subplots(figsize=(20, 7))

    pk1 = ax.bar(x + 0.00, pekerjaan[0], width, label='Kuliah', color='steelblue')
    pk2 = ax.bar(x + 0.15, pekerjaan[1], width, label='PNS', color='lightcoral')
    pk3 = ax.bar(x + 0.30, pekerjaan[2], width, label='Karyawan Swasta', color='blue')
    pk4 = ax.bar(x + 0.45, pekerjaan[3], width, label='Wirausaha', color='yellow')
    pk5 = ax.bar(x + 0.60, pekerjaan[4], width, label='Belum Bekerja', color='red')

    ax.set_title('Sebaran Pekerjaan Alumni Setelah Lulus', size=16)
    ax.set_ylabel('Jumlah', size=14)
    ax.set_xticks(x+0.3)
    ax.set_xticklabels(jurusan, size=10)
    ax.legend(fontsize=14)

    st.subheader('Sebaran Pekerjaan alumni SMKN 1 Kraksaan yang lulus tahun 2021 berdasarkan Jurusan Sekolah')
    st.pyplot()
    st.write('Disini dapat kita simpulkan bahwa memang seperti data global bahwa lulusan 2021 banyak yang belum bekerja.')
    st.write('Tetapi ada yang menarik di Jurusan Bisnis Daring dan Pemasaran bahwa alumni nya lebih banyak yang bekerja dibandingkan yang kuliah, dibandingkan dengan jurusan lain yang melanjutkan kuliah lebih banyak daripada yang bekerja')



page_names_to_funcs = {
    "Prekdiksi": prediksi,
    "Infografis": infografis
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()


