#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import holidays
import pickle

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns


# ## Laster inn rådata
# load_data() henter det tre rådatasetta fra mappen raw_data og legger det til variablane df_stations, df_trips, df_weather

# In[2]:


def load_data():
    df_stations = pd.read_csv("raw_data/stations.csv")
    df_trips = pd.read_csv("raw_data/trips.csv")
    df_weather = pd.read_csv("raw_data/weather.csv")

    model = pickle.load(open("model/rental_bikes_predictor.pkl", "rb"))

    print("Data lest in")

    return df_stations, df_trips, df_weather, model


# ## Funksjon for å rydde `stations.csv`
# 
# `stations_ready()` tar inn **stations.csv** og ryddar datasettet slik at det kan brukast vidare i modelleringa. Følgande steg blir gjennomført:
# 
# - **Filtrering:** Alle observasjonar frå *urelevante stasjonar* blir ekskludert.  
# - **Tidsstempel:** Kolonnen `"timestamp"` blir gjort om til eit *datetime-objekt*, slik at ein kan hente ut einingar som år, månad, dato og time.  
# - **Runding:** Alle ujamne tidsstempel blir *runda ned til næraste heile time*.  
# - **LOCF:** Det blir utført *forward-fill (ffill)* for å sikre kontinuerlege verdiar, der siste observasjon blir vidareført til neste måling.  
# - **Målvariabel:** Variabelen `"free_bikes_next"` blir lagt til ved å *flytte `"free_bikes"`-kolonnen eitt steg opp* i datasettet.  
#   - Før dette blir data *gruppert etter stasjon* for å unngå tidslekkasje mellom ulike stasjonar.
# 

# In[3]:


def stations_ready(df_stations):

    #Inkluderer bare relevante stasjoner
    stations = ['Møllendalsplass', 'Torgallmenningen', 'Grieghallen', 'Høyteknologisenteret', 
                'Studentboligene', 'Akvariet', 'Damsgårdsveien 71','Dreggsallmenningen Sør',
                'Florida Bybanestopp']
    df_stations = df_stations[df_stations["station"].isin(stations)]

    #Konverterer til datetime format og runder ned til hele timer
    df_stations['timestamp'] = pd.to_datetime(df_stations["timestamp"], utc=True, format='ISO8601')
    df_stations['timestamp'] = df_stations["timestamp"].dt.floor("h")

    #Sorterer etter stasjonsnavn og dato
    df_stations = df_stations.sort_values(["station", "timestamp"])

    #LOCF sist observert verdi av free_bikes og free_spots blir ført til neste time, fram til ein ny observasjon blir registrert
    df_stations = df_stations.set_index('timestamp').groupby('station')[['free_bikes', 'free_spots']].resample('h').last().ffill().reset_index()

    #Predikasjonsverdier: Flytter free_bikes ein time fram i tid ein rad bak
    df_stations["free_bikes_next"] = df_stations.groupby("station")["free_bikes"].shift(-1)

    return df_stations


# ## Funksjon for å rydde `trips.csv`
# 
# **`trips_ready()`** tar inn **`trips.csv`** og ryddar datasettet.  
# Følgande steg blir gjennomført:
# 
# - **Kolonner:** Unødvendige kolonner med *koordinatar* blir fjerna.  
# - **Tidsstempel:** Kolonnene **`started_at`** og **`ended_at`** blir gjort om til *datetime-objekt*.  
# - **Runding:** Alle tidsstempel blir *runda ned til næraste heile time*.  
# - **Aggregering:**  
#   - Talet på **avganger** blir rekna ut per stasjon og time (**`departures`**).  
#   - Talet på **ankomster** blir rekna ut per stasjon og time (**`arrivals`**).  
# - **Samanslåing:** Datasetta for *avganger* og *ankomster* blir slått saman til eitt felles datasett.  
# - **Manglande verdiar:** Timer utan avganger eller ankomster blir fylt med `0`.  
# - **Netto endring:** Kolonnen **`net_change`** blir lagt til som differansen mellom **`arrivals`** og **`departures`**.  

# In[4]:


def trips_ready(df_trips):

    #Dropper kolonnar
    cols = ["start_station_latitude", "start_station_longitude", "end_station_latitude", "end_station_longitude"]
    df_trips = df_trips.drop(columns=cols)

    #Konverterer til datetime og rundar ned til heile timer
    df_trips['started_at'] = pd.to_datetime(df_trips["started_at"], utc=True, format='ISO8601')
    df_trips['started_at'] = df_trips["started_at"].dt.floor("h")

    df_trips['ended_at'] = pd.to_datetime(df_trips["ended_at"], utc=True, format='ISO8601')
    df_trips['ended_at'] = df_trips["ended_at"].dt.floor("h")

    #Gjer om df_trips til eit df med features: {station, timestamp, departures, arrivals}
    departures = df_trips.groupby(["start_station_name", "started_at"]).size().reset_index(name="departures")
    departures = departures.rename(columns={"start_station_name": "station", "started_at": "timestamp"})

    arrivals = df_trips.groupby(["end_station_name", "ended_at"]).size().reset_index(name="arrivals")
    arrivals = arrivals.rename(columns={"end_station_name": "station", "ended_at": "timestamp"})

    df_trips = pd.merge(departures, arrivals, on=["station", "timestamp"], how="outer")

    #Fyller NaN verdier med 0
    df_trips = df_trips.fillna(0)

    #Legger til en ny kolonne {net_change}
    df_trips["net_change"] = df_trips["arrivals"] - df_trips["departures"]

    return df_trips


# ## Funksjon for å rydde `weather.csv`
# 
# **`weather_ready()`** tar inn **`weather.csv`** og ryddar datasettet.  
# Følgande steg blir gjennomført:
# 
# - **Manglande verdiar:** Alle numeriske kolonner blir behandla med *SimpleImputer* for å erstatte `NaN`-verdiar med *gjennomsnittet* for kvar kolonne.  
# - **Avrunding:** Etter imputering blir verdiane *avrunda til éin desimal*.  
# - **Tidsstempel:** Kolonnen **`timestamp`** blir gjort om til eit *datetime-objekt* slik at data kan knytast til same tidsoppløysing som stasjons- og turdata.  
# 

# In[5]:


def weather_ready(df_weather):

    #Initierer SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

    #Runder av desimalar til maks 1 desimalar
    numeric_cols = df_weather.select_dtypes(include=np.number).columns
    df_weather[numeric_cols] = np.round(imputer.fit_transform(df_weather[numeric_cols]), 1)

    #konverterer tids-data til datetime format
    df_weather["timestamp"] = pd.to_datetime(df_weather["timestamp"], utc=True, format="ISO8601")

    return df_weather


# ## Funksjon for å klargjere data
# 
# **`prep_data()`** tar inn dei tre rådatafilene — **`stations.csv`**, **`trips.csv`** og **`weather.csv`** — og køyrer kvar av dei gjennom sine funksjonar for datarydding.  
# Følgande steg blir gjennomført:
# 
# - **Stasjonsdata:** Blir behandla av **`stations_ready()`**
# - **Turdata:** Blir behandla av **`trips_ready()`** 
# - **Vêrdata:** Blir behandla av **`weather_ready()`**  
# - **Utskrift:** Funksjonen skriv ut `"Data ryddet"` når alle datasett er ferdig prosesserte.  
# 
# Returnerer dei tre rydda datasetta:  
# **`df_stations_ready`**, **`df_trips_ready`** og **`df_weather_ready`**.
# 

# In[6]:


def prep_data(df_stations, df_trips, df_weather):

    df_stations_ready = stations_ready(df_stations)
    df_trips_ready = trips_ready(df_trips)
    df_weather_ready = weather_ready(df_weather)

    print("\nData ryddet")

    return df_stations_ready, df_trips_ready, df_weather_ready


# ## Funksjon for å klargjere data til modellering
# 
# **`model_ready()`** kombinerer dei rydda datasetta frå **df_stations_ready**, **df_trips_ready** og **df_weather_ready** til eit samla datasett.
# Følgande steg blir gjennomført:
# 
# - **Merging:** Merger stasjons- og turdata basert på `station` og `timestamp`, og fyller manglande verdiar med `0`.  
# - **Tidsvariablar:** Legger til dato, time og sesongvariablar, samt sjekk for rushtid og fridagar*is_rush_hour* og *is_freeday* (raude dagar og helgedagar).  
# - **Vêrdata:** Slår saman vêrdata og fyller manglande verdiar ved *interpolering* og *forward/backward fill* for manglande verdier som evt ligger på enden av datasettet.  
# - **Opprydding:** Fjernar `timestamp` sidan denne blir delt opp i mindre einingar.  
# 
# Returnerer det ferdige datasettet **`df_model_ready`**, klart til visualisering og vidare feature engineering.
# 

# In[7]:


def model_ready(df_stations_ready, df_trips_ready, df_weather_ready):

    #Merger df_stations_ready og df_trips_ready og bruker bare radene til df_stations_ready slik at vi får berre samsvarande tidsdata
    df_model_ready = pd.merge(df_stations_ready, df_trips_ready, on=["station", "timestamp"], how="left")

    #Mergingen vil gi mange NaN verdier for heile timar der det ikkje er registrert noken ankomst/avgangar hos stasjonene, så vi fyller desse med 0
    df_model_ready = df_model_ready.fillna(0)

    #Henter ut norske raude dagar
    no_holidays = holidays.Norway(years=df_model_ready["timestamp"].dt.year.unique())

    #Legger til tid features
    df_model_ready = df_model_ready.assign(
    date = df_model_ready["timestamp"].dt.tz_localize(None).dt.normalize(),
    year = df_model_ready["timestamp"].dt.year,
    month = df_model_ready["timestamp"].dt.month,
    day = df_model_ready["timestamp"].dt.day,
    weekday = df_model_ready["timestamp"].dt.weekday,
    day_of_year = df_model_ready["timestamp"].dt.dayofyear,
    hour = df_model_ready["timestamp"].dt.hour,     
    is_rush_hour = df_model_ready["timestamp"].dt.hour.isin([6,7,8,9,15,16,17]).astype(int),
    is_freeday = df_model_ready["timestamp"].dt.weekday.isin([5, 6]).astype(int) | df_model_ready["timestamp"].dt.date.isin(no_holidays).astype(int),
    is_winter = df_model_ready["timestamp"].dt.month.isin([12, 1, 2]).astype(int),
    is_spring = df_model_ready["timestamp"].dt.month.isin([3, 4, 5]).astype(int),
    is_summer = df_model_ready["timestamp"].dt.month.isin([6, 7, 8]).astype(int),
    is_autumn = df_model_ready["timestamp"].dt.month.isin([9, 10, 11]).astype(int)
    )

    #merger inn værdata til model_ready
    df_model_ready = pd.merge(df_model_ready, df_weather_ready, on=["timestamp"], how="left")
    #Verdataen får manglande verdiar, så velger å interpolere desse med forwardfill og backwardfill dersom det er manglande verdier ved endane
    df_model_ready[["temperature","precipitation", "wind_speed"]] = df_model_ready[["temperature","precipitation", "wind_speed"]].interpolate(method='linear').ffill().bfill()

    #Droppar timestamp ettersom den er delt opp i mindre einingar
    df_model_ready = df_model_ready.drop(columns=["timestamp"])
    #Kan bruke model_ready til visualisering
    print("\nmodel_ready strukturert til visualisering")

    return df_model_ready


# ## Funksjon for å førebu data til modellering
# 
# **`prep_model_ready()`** tek inn **`df_model_ready`** og gjer vidare feature engineering på datasettet.  
# Følgande steg blir gjennomført:
# 
# - **Stasjonar:** Gjer om `station` til *dummyvariablar* slik at kvar stasjon får eigen binær kolonne.  
# - **Tidsmønster:** Konverterer tidsdata (`hour`, `weekday`, `day_of_year`) til *sinus og cosinusvariablar*.  
# - **Feature-reduksjon:** Fjernar kolonner som vil få samsvarande korrelasjon (`hour`, `day`, `month`, `year`, `weekday`, `date`).  
# 
# Returnerer det endelege datasettet **`df_model_ready`**, som no kan brukast til modelltrening.
# 

# In[8]:


def prep_model_ready(df_model_ready):

    #Gjør stasjoner til dummy variabler
    dummies = pd.get_dummies(df_model_ready["station"], prefix="station").astype(int)
    df_model_ready = pd.concat([df_model_ready.drop(columns=["station"]), dummies], axis=1)

    #Gjer tid til sinus og cosinus verdiar
    df_model_ready['hour_sin'] = np.sin(2 * np.pi * df_model_ready["hour"] / 24)
    df_model_ready['hour_cos'] = np.cos(2 * np.pi * df_model_ready["hour"] / 24)

    #Gjer dagar til sinus og cosinus verdiar
    df_model_ready['day_sin'] = np.sin(2 * np.pi * df_model_ready["weekday"] / 7)
    df_model_ready['day_cos'] = np.cos(2 * np.pi * df_model_ready["weekday"] / 7)

    #Gjer dagar i året til sinus og cosinus verdiar
    df_model_ready['dayofyear_sin'] = np.sin(2 * np.pi * df_model_ready["day_of_year"] / 365)
    df_model_ready['dayofyear_cos'] = np.cos(2 * np.pi * df_model_ready["day_of_year"] / 365)

    #Droppar features som vil få lik korrelasjon
    cols = ["hour", "day", "month", "year", "weekday", "date"]
    df_model_ready = df_model_ready.drop(columns=cols)

    #Datasettet er no klar til modellering
    print("\nmodel_ready ryddet til modellering")

    return df_model_ready


# ## Funksjon for å finne tidspunkt for prediksjon
# 
# **`find_prediction_time()`** identifiserer det siste registrerte tidspunktet i stasjonsdataene og bereknar kva tid neste prediksjon skal gjerast.  
# Følgande steg blir gjennomført:
# 
# - **Filtrering:** Beheld berre relevante stasjonar frå datasettet.  
# - **Tidsformat:** Konverterer kolonnen **`timestamp`** til *datetime*-objekt og gjer om tidssone til *Europe/Oslo*.  
# - **Siste registrering:** Finn det nyaste tidspunktet for kvar stasjon, og deretter det seinaste av alle.  
# - **Neste time:** Runder opp til næraste heile time og legg til éin time for å finne tidspunktet for neste prediksjon.  
# 
# Returnerer tre tidspunkt:  
# **`last_registered_time`**, **`next_hour`** og **`prediction_time`**.
# 

# In[9]:


def find_prediction_time(df_stations):
    #Inkluderer bare relevante stasjoner
    stations = ['Møllendalsplass', 'Torgallmenningen', 'Grieghallen', 'Høyteknologisenteret', 
                'Studentboligene', 'Akvariet', 'Damsgårdsveien 71','Dreggsallmenningen Sør',
                'Florida Bybanestopp']
    df_stations = df_stations[df_stations["station"].isin(stations)]

    #Gjer om timestamp kolonnen til datetime
    df_stations['timestamp'] = pd.to_datetime(df_stations["timestamp"], utc=True, format='ISO8601')

    #Konverterer til lokaltid Bergen
    df_stations['timestamp'] = df_stations['timestamp'].dt.tz_convert('Europe/Oslo')

    #Siste observasjon for kvar stasjon
    last_registered_times = df_stations.groupby("station")["timestamp"].max().reset_index()

    #Siste registerte tid for kvar stasjon
    last_registered_time = last_registered_times["timestamp"].max()

    #Nest heile time
    next_hour = last_registered_time.ceil("h")

    #Neste heile time til å predikere
    prediction_time = next_hour + pd.Timedelta(hours=1)

    return last_registered_time, next_hour, prediction_time


# ## Funksjon for å hente siste registrerte data før prediksjon
# 
# **`last_registered_times()`** Lagar model_ready på nytt og finner siste observasjon i datasettet for kvar stasjon
# Følgande steg blir gjennomført:
# 
# - **Datarydding:** Ryddar rådata igjen og lagar df_model_ready.
# - **Tidsberekning:** Brukar **`find_prediction_time()`** til å finne siste registrerte tidspunkt og tidspunktet for neste prediksjon.  
# - **Siste observasjonar:** Sorterer data etter dato og hentar siste rad per stasjon som representerer nyaste data.  
# - **Modellførebuing:** Gjer om siste observasjonar i df_model_ready til gyldige variablar til modellen med **`prep_model_ready()`**.  
# 
# Returnerer:
# - **`prepped_data`** – klargjort datasett til prediksjon  
# - **`last_registered_time`**, **`next_hour`**, **`prediction_time`** – tidspunkta for siste og neste observasjon  
# - **`df_last_registered_data`** – rådata for siste registrerte time per stasjon
# 

# In[10]:


def last_registered_times(df_stations, df_trips, df_weather):

    df_stations_ready, df_trips_ready, df_weather_ready = prep_data(df_stations, df_trips, df_weather)

    df_model_ready = model_ready(df_stations_ready, df_trips_ready, df_weather_ready)

    last_registered_time, next_hour, prediction_time = find_prediction_time(df_stations)

    df_last_registered_data = df_model_ready.sort_values('date').groupby('station').last().reset_index()

    prepped_data = prep_model_ready(df_last_registered_data)

    return prepped_data, last_registered_time, next_hour, prediction_time, df_last_registered_data


# ## Funksjon for å utføre prediksjonar
# 
# **`predict()`** brukar den ferdigtrente modellen til å estimere kor mange ledige sykler som vil vere tilgjengelege ved neste tidspunkt.  
# Følgande steg blir gjennomført:
# 
# - **Lagar X_predict:** Fjernar målvariabelen **`free_bikes_next`** frå datasettet for å bruke berre gyldige inputvariablar.  
# - **Prediksjon:** Brukar den modellen til å lage prediksjonar basert på **`prepped_data`**.  
# - **Avrunding:** Resultata blir avrunda til næraste "heile sykkel" for enklare tolkning.  
# 
# Returnerer ein numpy-array med dei predikerte verdiane for kvar stasjon.
# 

# In[11]:


def predict(prepped_data, model):

    X_predict = prepped_data.drop(columns=["free_bikes_next"])

    predictions = np.round(model.predict(X_predict), 0)

    return predictions


# ## Main funksjonen

# In[12]:


def main():

    df_stations, df_trips, df_weather, model = load_data()

    X, earliest_registered_time, next_hour, prediction_time, df_stations_predict = last_registered_times(df_stations, df_trips, df_weather)

    predictions = predict(X, model)

    station_pred_dict = dict(zip(df_stations_predict["station"], predictions))

    print(f"\nsiste tidsstempel i data {earliest_registered_time}")
    print(f"Neste hele time: {next_hour}")
    print(f"Time å predikere {prediction_time}")

    for _, row in df_stations_predict.iterrows():
        station_name = row['station']
        current_bikes = row['free_bikes']  # eller kolonnen som viser nåværende sykler
        predicted_bikes = station_pred_dict[station_name]
        print(f"Stasjon {station_name} : Nåværende {current_bikes} sykler, Predikert {predicted_bikes} sykler")




# In[13]:


if __name__ == "__main__":
    main()


# In[ ]:




