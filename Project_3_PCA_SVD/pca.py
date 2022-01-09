import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
os.chdir("C:\\Users\\Giorgos\\PycharmProjects\\pythonProject")
# Φόρτωση του συνόλου δεδομένων από το αρχείο "wdbc.data"
data = pd.read_csv("wdbc.data")

# Δημιουργία των τίτλων των στηλών του συνόλου δεδομένων
feature_names = ['id', 'type', 'mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error',
       'fractal dimension error', 'worst radius', 'worst texture',
       'worst perimeter', 'worst area', 'worst smoothness',
       'worst compactness', 'worst concavity', 'worst concave points',
       'worst symmetry', 'worst fractal dimension']
data.columns = feature_names

# Εξαίρεση των δύο πρώτων στηλών του συνόλου,
# δεδομένου ότι δεν περιλαμβάνουν τιμές χαρακτηριστικών
# (id της μέτρησης και ένδειξη κακοήθους (M) και καλοήθους (B)
x = data.drop(['id'], axis=1)
x = x.drop(['type'], axis=1)

# Κανονικοποίηση των τιμών των χαρακτηριστικών
# του συνόλου των δεδομένων
x = StandardScaler().fit_transform(x) # normalizing the features
print (np.mean(x),np.std(x))

# Προβολή στο επίπεδο των 2 κύριων συνιστωσών
pca_data = PCA(n_components=2)
PC = pca_data.fit_transform(x)
print(PC)

# Δημιουργία DataFrame με τις κύριες συνιστώσες για όλες τις μετρήσεις
PC_df = pd.DataFrame(data = PC, columns = ['PC 1', 'PC 2'])

# Εμφάνιση ποσοστού των αρχικών δεδομένων που διατηρούνται στις κύριες συνιστώσες
print('Percent of data held per principal component: ' + str(pca_data.explained_variance_ratio_))

# Διάγραμμα των κύριων συνιστωσών
plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Κύρια συνιστώσα - 1',fontsize=20)
plt.ylabel('Κύρια συνιστώσα - 2',fontsize=20)
plt.title("Ανάλυση Κύριων Συνιστωσών \nγια δεδομένα καρκίνου του μαστού",fontsize=20)
targets = ['B', 'M']
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = data['type'] == target
    plt.scatter(PC_df.loc[indicesToKeep, 'PC 1']
               , PC_df.loc[indicesToKeep, 'PC 2'], c = color, s = 50)

plt.legend(targets,prop={'size': 15})
plt.show()
