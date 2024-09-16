# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 13:02:41 2024

@author: Manu2
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import sample

import seaborn as sns

#Importation des données
carac=pd.read_csv(filepath_or_buffer="C:\\Users\\Manu2\\OneDrive\\Bureau\\Projet DataScientest\\Donnee\\caracteristics.csv",sep=",",encoding='latin-1')
lieux=pd.read_csv(filepath_or_buffer="C:\\Users\\Manu2\\OneDrive\\Bureau\\Projet DataScientest\\Donnee\\places.csv",sep=",",encoding='latin-1')
user=pd.read_csv(filepath_or_buffer="C:\\Users\\Manu2\\OneDrive\\Bureau\\Projet DataScientest\\Donnee\\users.csv",sep=",",encoding='latin-1')
car=pd.read_csv(filepath_or_buffer="C:\\Users\\Manu2\\OneDrive\\Bureau\\Projet DataScientest\\Donnee\\vehicles.csv",sep=",",encoding='latin-1')
population=pd.read_csv(filepath_or_buffer="C:\\Users\\Manu2\\OneDrive\\Bureau\\Projet DataScientest\\Donnee\\population.csv",sep=',')


carac.dep=carac.dep.astype("object")
population.dep=population.dep.astype("object")
carac = pd.merge(carac, population, on='dep', how='left')

carac.describe()

user.sexe.value_counts(normalize=True)

#caractéristique
def inconnu(dataframe):
  """
  Remplace toutes les occurrences de -1 par NaN dans un DataFrame.
  Returns:
    Le DataFrame avec les -1 remplacés par NaN.
  """
  
  # Remplacer toutes les valeurs -1 par NaN dans le DataFrame
  dataframe = dataframe.replace(-1, np.nan)
  
  return dataframe
    

#Partie exploratoire des données

#graphique pourcentage des valeurs manquantes
liste=[carac,lieux,car,user]
# Calcul du pourcentage de valeurs manquantes par colonne
for i in liste:
    missing_percent = (i.isnull().sum() / len(carac)) * 100
    sns.barplot(x=missing_percent.index, y=missing_percent)
    plt.xlabel('Variables')
    plt.ylabel('Pourcentage de valeurs manquantes')
    plt.title('Pourcentage de valeurs manquantes par variable')
    plt.xticks(rotation=45)
    plt.show()
    
#Graphique gravité de l'accident par sexe
sns.countplot(y="grav",hue='sexe',data=user)
plt.title('Nombre d\'accidents par gravité et par genre')
plt.ylabel('Nombre d\'accidents')
plt.show()

sns.countplot(x='grav', data=user)
plt.title('Distribution de la gravité des accidents par âge')
plt.xlabel('gravité')
plt.ylabel('Nombre d\'accidents')
plt.legend(title='Gravité des accidents')




user.info()

"""Les variables à haut pourcentage de valeur manquante ne seront pas gardé"""
carac.info()

liste=[carac,lieux,car,user]

#Pour transformer les float en int et les na en -1 qui seront retransformer plus tard en -1
#Pour NA 
for df in liste:
    for col in df:
        if df[col].dtype == 'float64':
            df[col] = df[col].fillna(-1)
            df[col]=df[col].astype(int)
    print(df.info())

#regroupement de modalité de certaines variables qui ont trop de modalité 
#pour éviter de rendre le modèle de machine learning plus complexe à entraîner et à interpréter
#et éviter le surapprentissage 
#Dimensionnalité : Chaque modalité supplémentaire correspond à une nouvelle dimension dans l'espace 
#des caractéristiques. Un grand nombre de dimensions peut rendre l'algorithme de machine learning moins efficace et plus sensible au bruit.
#Pour facilité l'interprétation (homogénéité des groupes)

def pop_regroupement(population):
    if 200000 < population:
        return "very_small_pop"
    if 200000 <= population < 500000:
        return "small_pop"
    if 500000 <= population < 1000000:
        return "medium_pop"
    if 1000000 <= population < 2000000:
        return "big_pop"
    if population >= 2000000:
        return "very_big_pop"
    
carac['reg_pop'] = carac['population'].apply(pop_regroupement)

#regroupement intersection
def int_regroupement(x):

  int_reg = {
      (2, 3, 4, 5): 1
  }

  for int_tuple, int_nom in int_reg.items():
      if x in int_tuple:
          return int_nom
      else:
          return x

carac['int_reg'] = carac['int'].apply(int_regroupement)

#regroupement des conditions atmosphériques
def atm_regroupement(atm):

  atm_reg = {
      (2,3): 1
  }

  for atm_tuple, atm_nom in atm_reg.items():
      if atm in atm_tuple:
          return atm_nom
      else:
          return atm

carac['atm_reg'] = carac['atm'].apply(atm_regroupement)


#regroupement du type de collision
def col_regroupement(x):

  col_reg = {
      (1,2,3): 1,
      (4,5): 2
  }

  for col_tuple, col_nom in col_reg.items():
      if x in col_tuple:
          return col_nom
      else:
          return x

carac['col_reg'] = carac['col'].apply(col_regroupement)


#regroupement des mois en saison
def determiner_saison(mois):
  """Détermine la saison en fonction du numéro du mois.

  Args:
    mois: Le numéro du mois (1-12).

  Returns:
    La saison correspondante (hiver, printemps, été, automne).
  """

  saisons = {
      (1, 2, 12): 'hiver',
      (3, 4, 5): 'printemps',
      (6, 7, 8): 'été',
      (9, 10, 11): 'automne'
  }

  for saison_tuple, saison_nom in saisons.items():
      if mois in saison_tuple:
          return saison_nom

  return "Saison inconnue"


carac['saison'] = carac['mois'].apply(determiner_saison)

carac['an']=carac['an']+2000

carac[['an','mois','jour']]=carac[['an','mois','jour']].astype('str')
carac['date']=carac['an']+'-'+carac['mois']+'-'+carac['jour']
carac['date']=pd.to_datetime(carac['date'])
carac['weekday']=carac['date'].dt.day_name()

#fonction pour la période de la semaine
def determiner_jour(weekday):

  periode = {
      ('Monday','Tuesday','Friday','Wednesday','Thursday'): "week",
      ('Sunday','Saturday'): "weekend"
  }

  for periode_tuple, periode_nom in periode.items():
      if weekday in periode_tuple:
          return periode_nom

  return

carac['periode'] = carac['weekday'].apply(determiner_jour)

carac.info()

#fonction pour créer les tranches horaires

def tranche_horaire(heure):
    if 500 <= heure < 1200:
        return "morning"
    if 1200 <= heure < 1700:
        return "afternoon"
    if 1700 <= heure < 2300:
        return "evening"
    if 0 <= heure < 500:
        return "night"
    
carac['tranche_horaire'] = carac['hrmn'].apply(tranche_horaire)

counts_horaire = carac['tranche_horaire'].value_counts()

# Pie selon la tranche horaire
plt.pie(counts_horaire, labels=counts_horaire.index, autopct='%1.1f%%', startangle=90)
plt.title("Répartition par tranche horaire")
plt.show()

#jointure pour l'age

#Usagers
user.info()

unique_secu=user.secu.value_counts()

#regroupement par type de sécurité
def regroupement_secu(secu):
    secu_reg = {
        (11,21,31,41,91): 1,
        (12,22,32,42,92):2,
        (0,1,2,3,10,13,20,23,30,33,40,43,90,93):3
    }

    for secu_tuple, secu_nom in secu_reg.items():
        if secu in secu_tuple:
            return secu_nom


user['secu_reg']=user['secu'].apply(regroupement_secu)

carac[['Num_Acc','an']]=carac[['Num_Acc','an']].astype('int64')

user=pd.merge(user,carac,left_on='Num_Acc',right_on='Num_Acc')

user.an=user.an.astype('int')

user['age']=user.an-user.an_nais

#tranche d'âge
def tranche_age(age):
    if 0 <= age <= 10:
        return "age_0_10"
    if 10 < age <= 20:
        return "age_10_20"
    if 20 < age <= 30:
        return "age_20_30"
    if 30 < age <= 40:
        return "age_30_40"
    if 40 < age <= 50:
        return "age_40_50"
    if 50 < age <= 60:
        return "age_50_60"
    if 60 < age <= 70:
        return "age_60_70"
    if 70 < age <= 80:
        return "age_70_80"
    if 90 < age <= 100:
        return "age_90_100"
    
user['tranche_age'] = user['age'].apply(tranche_age)

#Graphique implicant l'âge maintenant que la variable âge est créé
bins = [0, 10, 20, 30, 40, 50, 60, 70, 100]
user['groupe_age'] = pd.cut(user['age'], bins=bins)

plt.figure(figsize=(10, 6))
sns.countplot(x='groupe_age', hue='grav', data=user, palette='deep')
plt.title('Distribution de la gravité des accidents par âge')
plt.xlabel("Groupe d'âge")
plt.ylabel('Nombre d\'accidents')
plt.legend(title='Gravité par âge')

# Deuxième graphique : Distribution globale
plt.figure(figsize=(8, 4))
sns.countplot(x='grav', data=user)
plt.title('Distribution globale de la gravité des accidents')
plt.xlabel('Gravité')
plt.ylabel('Nombre d\'accidents')
plt.legend().remove()

plt.show()

#Gravité des accidents hors et en agglomération
sns.countplot(x='grav', hue='agg', data=user, palette='deep')
plt.title('Distribution de la gravité des accidents par agg')
plt.xlabel("grav")
plt.ylabel('agg')
plt.legend(title='Gravité par agg')


user=user.drop(["an_nais","mois","date","jour","hrmn","int","atm","col","adr","gps","lat","long","dep","secu","population"],axis=1)


#Base véhicule

#regroupement de catégorie de véhicule
def regroupement_vehicule(catv):
    vehicules = {
        (1,2,4,5,6,30,31,32,33,34,41,42,43,80,50,60,80): 2,
        (3,7,8,9,35,36): 1,
        (10,11,12,13,14,15,16,17,20,21): 3,
        (37,38,39,40,18,19): 4,
        (0) : 5,
        (99): 6
    }

    for vehicules_tuple, vehicules_nom in vehicules.items():
        if catv in vehicules_tuple:
            return vehicules_nom
        else:
            return catv

car['voiture_reg']=car['catv'].apply(regroupement_vehicule)

#regroupement de la variable obstacle fixe heurté
def regroupement_obs(obs):
    obs_reg = {
        (12,11,16): 1,
        (3,4,5,7): 2,
        (2,13,17): 3,
        (6,10,8,9): 4,
        (1,14,15) : 5,
        (0): 6
    }

    for obs_tuple, obs_nom in obs_reg.items():
        if obs in obs_tuple:
            return obs_nom
        else:
            return obs

car['obs_reg']=car['obs'].apply(regroupement_obs)


#Regroupement de la variable type de choc
def regroupement_choc(choc):
    choc_reg = {
        (1,2,3): 1,
        (4,5,6): 2,
        (7,8): 3,
        (9): 4,
        (0): 0
    }

    for choc_tuple, choc_nom in choc_reg.items():
        if choc in choc_tuple:
            return choc_nom
        else:
            return choc

car['choc_reg']=car['choc'].apply(regroupement_choc)



#type de manoeuvre
def regroupement_manv(manv):
    manv_reg = {
        (1,2): 1,
        (11,12): 2,
        (17,18): 3,
        (13,14): 4,
        (15,16): 5,
        (3,6,7,8,9,5):6,
        (10):7,
        (4,20,24,23):8,
        (21,22,26,19,25):9
    }

    for manv_tuple, manv_nom in manv_reg.items():
        if manv in manv_tuple:
            return manv_nom
        else:
            return manv
        
car['manv_reg']=car['manv'].apply(regroupement_manv)

car['senc'] = car['senc'].replace(0, -1)

car=car.drop(["choc","obs","catv","manv"],axis=1)

car=car.apply(inconnu)

#Base usagers
user.info()


def regroupement_trajet(trajet):
    trajet_reg = {
        (1,2): 1,
        (3,5): 2,
        (4): 3,
        (9): 4,
        (0,-1):-1
    }

    for trajet_tuple, trajet_nom in trajet_reg.items():
        if trajet in trajet_tuple:
            return trajet_nom
        else:
            return trajet

user['trajet_reg']=user['trajet'].apply(regroupement_trajet)


user=user.drop(["age","trajet"],axis=1)

#Lieux 
lieux.info()

lieux=lieux.apply(inconnu)

lieux=lieux.drop(["voie","v1","v2","pr","pr1"],axis=1)

liste=[carac,lieux,car,user]

#Pour transformer les float en int et les na en -1 qui seront retransformer plus tard en -1
#Pour NA 
for df in liste:
    for col in df:
        if df[col].dtype == 'float64':
            df[col] = df[col].fillna(-1)
            df[col]=df[col].astype(int)
    print(df.info())

#Matrice des corrélations de la base usager
#user2=user.drop(['Num_Acc','num_veh','com','an','groupe_age'],axis=1)

#user2=user2.drop(['saison','weekday','periode','tranche_horaire','tranche_age'],axis=1)


#user_cat=user[['saison','weekday','periode','tranche_horaire','tranche_age']]
#user_encoded=pd.get_dummies(user_cat)
#user_concat=pd.concat([user2, user_encoded], axis=1)
#corr_matrix = user_concat.corr()

# Créer la carte thermique
#sns.set(rc={'figure.figsize': (30, 20)})  # Width: 12 inches, Height: 8 inches

# Create your heatmap with the same parameters
#sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
#            linewidths=.5, annot_kws={"size": 15})

#plt.xticks(rotation=90)
#plt.yticks(rotation=0)
#plt.show()


#Jointure des variables
user=user.drop_duplicates(subset=["Num_Acc","place","num_veh","grav"])

base_merge=pd.merge(user,lieux,left_on='Num_Acc',right_on='Num_Acc')
#base_merge=pd.merge(base_merge,user,left_on='Num_Acc',right_on='Num_Acc')
base_merge=pd.merge(base_merge,car,left_on=['Num_Acc','num_veh'],right_on=['Num_Acc','num_veh'])

#vérification de doublons
base_merge= base_merge.drop_duplicates()

base_merge=base_merge.drop(['Num_Acc','num_veh','com','groupe_age'],axis=1)

base_merge.info()

#échantillonage par stratification
proportions = base_merge['grav'].value_counts(normalize=True)

base_merge=base_merge.apply(inconnu)

df_sample = base_merge.groupby('grav').sample(frac=0.05, replace=False,random_state=np.random.RandomState(42))
proportions_sample = df_sample['grav'].value_counts(normalize=True)

#Préparation données pour modélisation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix


df_sample=df_sample.astype('object')
df_sample.info()

target = df_sample["grav"]
feats = df_sample.drop(["grav"],axis=1)

X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.25, random_state=43)

#X_train
X_train_num=X_train[["occutc","larrout","lartpc","nbv"]]
X_train_num=X_train_num.astype("float")

X_train_cat = X_train.drop(X_train_num.columns, axis=1)

#X_test
X_test_num=X_test[["occutc","larrout","lartpc","nbv"]]
X_test_num=X_test_num.astype("float")

X_test_cat = X_test.drop(X_test_num.columns, axis=1)


# Appliquer l'imputation
imputer=SimpleImputer(missing_values=np.nan, strategy='median')
X_train_num_imputed=imputer.fit_transform(X_train_num)
X_test_num_imputed=imputer.transform(X_test_num)

imputer=SimpleImputer(missing_values=np.nan, strategy='most_frequent')
X_train_cat_imputed=imputer.fit_transform(X_train_cat)
X_test_cat_imputed=imputer.transform(X_test_cat)

#Standardisation des données
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train_num_imputed)
X_test_scaled=scaler.transform(X_test_num_imputed)

df1 = pd.DataFrame(X_train_scaled)
df2 = pd.DataFrame(X_test_scaled)
df3 = pd.DataFrame(X_train_cat_imputed)
df4 = pd.DataFrame(X_test_cat_imputed)

#Pour avoir des noms de variables compréhensible
X_train_num_names = dict(zip(df1.columns, X_train_num.columns))
X_train_scaled = df1.rename(columns=X_train_num_names)

X_test_num_names = dict(zip(df2.columns, X_test_num.columns))
X_test_scaled = df2.rename(columns=X_test_num_names)

X_train_cat_names = dict(zip(df3.columns, X_train_cat.columns))
X_train_cat_imputed = df3.rename(columns=X_train_cat_names)

X_test_cat_names = dict(zip(df4.columns, X_test_cat.columns))
X_test_cat_imputed = df4.rename(columns=X_test_cat_names)

encoder = OneHotEncoder()
X_train_encoded = encoder.fit_transform(X_train_cat_imputed)

# Obtention des noms des nouvelles colonnes
new_columns_train = encoder.get_feature_names_out()

# Création d'un nouveau DataFrame avec les noms de colonnes
X_train_encoded = pd.DataFrame(X_train_encoded.toarray(), columns=new_columns_train)

X_test_encoded = encoder.transform(X_test_cat_imputed)

# Obtention des noms des nouvelles colonnes
new_columns_test = encoder.get_feature_names_out()

# Création d'un nouveau DataFrame avec les noms de colonnes
X_test_encoded = pd.DataFrame(X_test_encoded.toarray(), columns=new_columns_test)

X_train_reconstitue=pd.concat([X_train_scaled, X_train_encoded], axis=1)
X_test_reconstitue=pd.concat([X_test_scaled, X_test_encoded], axis=1)

#X_train_reconstitue=X_train_reconstitue.drop(["voiture_reg_8"],axis=1)

y_train_clf = pd.Categorical(y_train)

clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrainement du modèle
clf.fit(X_train_reconstitue,y_train_clf)

# Prédiction
predictions = clf.predict(X_test_reconstitue)

#y_test_clf=pd.Categorical(y_test)

y_train_2 = y_train.astype(int)
y_test_2 = y_test.astype(int)

# Métrique du modèle

confusion_mat = confusion_matrix(y_test_2, predictions)
print("Confusion Matrix:\n", confusion_mat)

accuracy = accuracy_score(y_test_2, predictions)
print("Accuracy:", accuracy)

print("Score du modèle sur le modèle d'entrainement:",clf.score(X_train_reconstitue,y_train_2))
print("Score du modèle sur le modèle de test:",clf.score(X_test_reconstitue,y_test_2))

# Obtenir l'importance des caractéristiques
importances = clf.feature_importances_

# Afficher les résultats

feat_importances = pd.DataFrame(clf.feature_importances_, index=X_train_reconstitue.columns, columns=["Importance"])
feat_importances.sort_values(by='Importance', ascending=False, inplace=True)

#Graphique d'importance
feat_importances.plot(kind='bar', figsize=(8,6))

# Créer un DataFrame pour mieux visualiser
feature_importances = pd.Series(importances, index=X_train_reconstitue.columns)
feature_importances = feature_importances.sort_values(ascending=False)

# Afficher les 20 premières features
print(feature_importances[:20])
X_train_reconstitue.info()

# Visualiser les résultats
feature_importances[:20].plot(kind='barh')
plt.xlabel('Importance')
plt.ylabel('Caractéristiques')
plt.title('Importance des 20 premières caractéristiques')
plt.show()

#Modélisation avec 20 variables
new_feat=feature_importances[:30]
new_feat
#Récupération des variables dans la base d'échantillonage
liste_feat=new_feat.index.tolist()

X_train_important=X_train_reconstitue[liste_feat]
X_test_important=X_test_reconstitue[liste_feat]
new_feat
#modèle
y_train_clf = pd.Categorical(y_train)

clf = RandomForestClassifier(n_estimators=100, random_state=42)

clf.fit(X_train_important,y_train_clf)

predictions = clf.predict(X_test_important)

y_train_2 = y_train.astype(int)
y_test_2 = y_test.astype(int)

print("Accuracy score Random Forest base d'entrainement:",clf.score(X_train_important,y_train_2))
print("Accuracy score Random Forest base de test:",clf.score(X_test_important,y_test_2))

#Toujours en sur-apprentissage

from sklearn.linear_model import LogisticRegression

model_Regression = LogisticRegression()
model_Regression.fit(X_train_reconstitue, y_train_clf)

y_pred_LR = model_Regression.predict(X_test_reconstitue)

# Matrice de confusion
confusion_matrix(y_test_2, y_pred_LR)

# Accuracy
print("Score du modèle sur la base de test RegLogistisque:",accuracy_score(y_test_2, y_pred_LR))
print("Score du modèle sur la base d'entrainement RegLogistisque:",model_Regression.score(X_train_reconstitue,y_train_2))

#------------------------------------------------------------------------------------------------------------------------------

#Oversampling

from imblearn.over_sampling import RandomOverSampler

rOs = RandomOverSampler()
X_ro, y_ro = rOs.fit_resample(X_train_reconstitue, y_train_2)

# Entraînement du modèle de régression logistique
lr = LogisticRegression()
lr.fit(X_ro, y_ro)
# Affichage des résultats
y_pred_OverSampling = lr.predict(X_test_reconstitue)

print("Score du modèle sur la base d'entrainement RegLogistisque Over sampling:",model_Regression.score(X_ro,y_ro))
print("Score du modèle sur la base de test RegLogistisque Over sampling:",accuracy_score(y_test_2, y_pred_OverSampling))
#Moins performant

confusion_matrix(y_test_2, y_pred_OverSampling)

#
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrainement du modèle
clf.fit(X_ro,y_ro)

# Prédiction
y_pred_OverSampling_clf = clf.predict(X_test_reconstitue)

#y_test_clf=pd.Categorical(y_test)

y_train_2 = y_train.astype(int)
y_test_2 = y_test.astype(int)

# Métrique du modèle

confusion_mat = confusion_matrix(y_test_2, y_pred_OverSampling_clf)
print("Confusion Matrix:\n", confusion_mat)

accuracy = accuracy_score(y_test_2, y_pred_OverSampling_clf)
print("Accuracy test après oversampling:", accuracy)

print("Score du modèle sur le modèle d'entrainement:",clf.score(X_ro,y_ro))
#Sur-apprentissage

#------------------------------------------------------------------------------------------------------
#H2O

#Parti à lancé de façon optionnel
'''
import h2o
from h2o.automl import H2OAutoML
h2o.init()

y_train.info()


aml = H2OAutoML(max_models=10, seed=1)
X_train_to_h2o=pd.concat([X_train, y_train], axis=1)
replace_dict = {'1': 'Indemne', '2': 'Tué', '3':'Blesse_hospitalise','4':'Blesse leger'}
X_train_to_h2o['grav'] = X_train_to_h2o['grav'].replace(replace_dict)
X_train_h2o=h2o.H2OFrame(X_train_to_h2o)

#aml.train(x=list(X_train.columns), y="grav", training_frame=X_train_h2o)



# Diviser les données en ensemble d'entraînement et de test
train, test = X_train_h2o.split_frame(ratios=[0.8])

# Créer un objet AutoML
aml = H2OAutoML(max_models=20)

# Entraîner le modèle
aml.train(y="grav", training_frame=train)

# Obtenir le meilleur modèle
lb = aml.leaderboard
lb.head(rows=5)

# Faire des prédictions sur l'ensemble de test
preds = aml.leader.predict(test)

best_model = aml.leader
# Évaluer les performances du modèle
performance = best_model.model_performance(test)

print(best_model)
'''