import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Memuat dataset diabetes dari file CSV
data = pd.read_csv('diabetes.csv')

X = data.drop('Outcome', axis=1) # Membuat data prediktor (fitur) dengan menghapus kolom 'Outcome'
y = data['Outcome']

# Memisahkan data menjadi dataset pelatihan (80%) dan pengujian (20%).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = KNeighborsClassifier()
model.fit(X_train, y_train)

 # Menghitung akurasi model menggunakan dataset pengujian
score = model.score(X_test, y_test)
print(f'Model score: {score}')

# Menyimpan model yang telah dilatih
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Menyimpan data hasil pengujian (X_test dan y_test)
X_test.to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

