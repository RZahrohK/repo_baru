import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

lokasi_file=input("Masukkan lokasi file data training : ")
lokasi_file="Small Training Set.csv"
data_train= pd.read_csv(lokasi_file)
#Mengambil dataset dari atribut selain status sebagai data atribut fitur
#dan dari data status sebagai atribut target
x,y = data_train.loc[:,data_train.columns != 'status'], data_train.loc[:,'status']
#Mengambil data kolom atiribut fitur
col=data_train.columns.tolist()
col.remove('status')


#Membangun KNN Classifier dengan n mulai dari 1 hingga - n sesuai masukan pengguna
n=int(input("Masukkan nilai N maksimal yang anda inginkan : "))
knn = KNeighborsClassifier(n_neighbors = n)
#Melakukan enccoding pada atribut yang berisi string.
x=pd.get_dummies(x, columns=col)
y=pd.get_dummies(y, columns=['status'])

# membangun traning set, dimana 80% utk training dan 20% utk testing
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
neig = np.arange(1,n)
train_accuracy = []
test_accuracy = []
for i, k in enumerate (neig):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train,y_train)
    train_accuracy.append(knn.score(x_train,y_train))
    test_accuracy.append(knn.score(x_test,y_test))

# Plot
plt.figure(figsize=(13,8))
plt.plot(neig, test_accuracy, label = 'Akurasi IDS berbasis KNN')
plt.legend()
plt.title('AKURASI')
plt.xlabel('Jumlah K')
plt.ylabel('Akurasi')
plt.xticks(neig)
plt.show()




