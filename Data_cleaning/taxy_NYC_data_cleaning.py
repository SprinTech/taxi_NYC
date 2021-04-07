# Import des librairies
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score

# Transformation du dataset en dataframe
taxi = pd.read_csv('Data/01_raw/train.csv')

# Sélection des colonnes
df = taxi.drop(['store_and_fwd_flag','passenger_count'], axis= 1)

# Traitement des données de type date
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], format="%Y-%m-%d %H:%M:%S")
df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'], format="%Y-%m-%d %H:%M:%S")
df['day_of_week'] = df['pickup_datetime'].dt.day_name()

# Calcul de la distance
def haversine_vectorize(lon1, lat1, lon2, lat2):
    """ return the distance in km etween two point with longitude and latitude defined"""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    newlon = lon2 - lon1
    newlat = lat2 - lat1

    haver_formula = np.sin(newlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(newlon/2.0)**2

    dist = 2 * np.arcsin(np.sqrt(haver_formula ))
    km = 6367 * dist #6367 for distance in KM for miles use 3958
    return km
    
df['distance'] = round(haversine_vectorize(df['pickup_longitude'],df['pickup_latitude'],df['dropoff_longitude'],df['dropoff_latitude']),2)

# Encodage du jour de la semaine
day_df = pd.DataFrame(df, columns=['day_of_week'])
dum_df = pd.get_dummies(day_df, columns=["day_of_week"], prefix=["day_is"] )
df = df.join(dum_df)


# Calcul de la vitesse moyenne
def vitesse(d,t):
    v= round((d*3600)/t,2)
    return v

df['vitesse']=vitesse(df['distance'],df['trip_duration'])

# Exclusion des valeurs aberrantes
df = df.drop(df[df['trip_duration'] > 10000].index)
df = df.drop(df[(df['distance'] < 1)|(df['distance'] > 40)].index)
df = df.drop(df[(df['vitesse'] > 60) | (df['vitesse'] < 10)].index)

# Ajout de la tranche horaire
df['time_slot'] = df['pickup_datetime'].apply(lambda x : "0-3h" if x.hour < 4 
                                                else "4-7h" if x.hour < 8 
                                                else "8-11h" if x.hour < 12 
                                                else "12-15h" if x.hour < 16 
                                                else "16-19h" if x.hour < 20 
                                                else "20-23h")

df.to_csv('cleaned_data.csv', index = False)

