#!/usr/bin/env python
# coding: utf-8

# ## Laster inn biblioteker

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


# ## Laster inn rådata
# load_data() henter det tre rådatasetta fra mappen raw_data og legger det til variablane df_stations, df_trips, df_weather

# In[2]:


def load_data():
    df_stations = pd.read_csv("raw_data/stations.csv")
    df_trips = pd.read_csv("raw_data/trips.csv")
    df_weather = pd.read_csv("raw_data/weather.csv")

    print("Data lest in")

    return df_stations, df_trips, df_weather


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


# ## Funksjon for å dele data til modellering
# 
# **`train_val_test()`** deler det ferdige datasettet **`df_model_ready`** inn i trenings, validerings og testsett.  
# Følgande steg blir gjennomført:
# 
# - **Datasplitting:**  
#   - 70 % av dataen blir brukt til *trening* (**`df_train`**)  
#   - 15 % til *validering* (**`df_val`**)  
#   - 15 % til *testing* (**`df_test`**)  
# - **Rekkefølge:** Splittinga beheld den opprinnelege tidsrekkefølga, noko som er viktig for tidsseriedata.  
# 
# Returnerer dei tre datasetta: **`df_train`**, **`df_val`** og **`df_test`**.
# 

# In[9]:


def train_val_test(df_model_ready):
    #Splitter opp i trenings, testing- og valideringsdata
    n = len(df_model_ready)
    train_end = int(n * 0.7)   # 70% train
    val_end   = int(n * 0.85)  # 15% validation, 15% test

    df_train = df_model_ready.iloc[:train_end]
    df_val   = df_model_ready.iloc[train_end:val_end]
    df_test  = df_model_ready.iloc[val_end:]

    print("\nModellerings-data oppdelt")

    return df_train, df_val, df_test


# ## Funksjon for hyperparametertuning
# 
# **`tune_hyperparameters()`** justerer modellens hyperparametrar ved bruk av *GridSearchCV*.  
# Følgande steg blir gjennomført:
# 
# - **Scoring:** Brukar metrikkane *RMSE*, *MAE* og *R²* for modellvurdering.  
# - **Grid search:** Utfører *kryssvalidering med 5 folds (cv=5)* for å finne den kombinasjonen av parametre som gir lågaste *RMSE*.  
# - **Resultat:** Hentar ut den beste modellen (**`best_model`**) og bereknar *RMSE* på treningsdata.  
# 
# Returnerer **`best_model`** og objektet **`grid_search`** med resultata frå søket.
# 

# In[10]:


def tune_hyperparameters(model, param, X, y):

    #Scoringar for å evaluere modellar
    scoring = {'RMSE': 'neg_root_mean_squared_error', 'MAE': 'neg_mean_absolute_error','R2': 'r2'}

    grid_search = GridSearchCV(model, param_grid=param, cv=5, scoring=scoring, refit='RMSE', n_jobs=-1)
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    rmse_train = np.sqrt(mean_squared_error(y, best_model.predict(X)))

    print(f"\nRMSE på treningsdata: RMSE={rmse_train:.3f})")

    return best_model, grid_search


# ## Funksjon for å trene og samanlikne modellar
# 
# **`train_models()`** trenar og evaluerer fleire ulike modellar for å finne den med best predikasjonsevne.  
# Følgande steg blir gjennomført:
# 
# - **Datasplitting:** Deler treningsdata i features (**X**) og målvariabel (**y**) for både trening og validering.  
# - **Modellar:** Fem modellar blir testa — *DummyRegressor*, *Lasso*, *LightGBM*, *CatBoost* og *XGBoost*.  
# - **Hyperparametertuning:** Det blir gjennomført hyperparameter justeringar med *GridSearchCV* gjennom funksjonen **`tune_hyperparameters()`**. Dette blir ikkje brukt på baseline modellen  
# - **Evaluering:** *RMSE* blir brukt som evalueringsmål på både trenings- og valideringsdata.  
# - **Utvelging:** Den modellen som får lågaste *RMSE* på valideringsdata blir valt som *beste modell*.  
# - **Testing:** Beste modell blir sendt vidare til **`train_best_model()`** for testing på testsettet.  
# 
# Returnerer den ferdigtrente **`best_model`**.
# 

# In[11]:


def train_models(df_train, df_val, df_test):
    #Deler opp i X og y variablar for trening og validering
    X_train = df_train.drop(columns=["free_bikes_next"])
    X_val = df_val.drop(columns=["free_bikes_next"])

    y_train = df_train["free_bikes_next"]
    y_val = df_val["free_bikes_next"]

    #Modellar som skal trenast
    models = {
    "DummyRegressor": DummyRegressor(strategy="mean"),
    "Lasso": Lasso(random_state=0),
    "LightGBM": LGBMRegressor(objective='regression', random_state=42, verbose=-1),
    "CatBoost": CatBoostRegressor(verbose=0, random_seed=42, loss_function='RMSE'),
    "XGB": XGBRegressor(tree_method='hist', enable_categorical=True, random_state=42)
        }

    rmse_scores = {}

    for name, model in models.items():

        if name == "DummyRegressor":

            best_model = model.fit(X_train, y_train)
            prediction = best_model.predict(X_val)
            rmse_train = np.sqrt(mean_squared_error(y_train, best_model.predict(X_train)))
            print(f"Baseline RMSE på treningsdata: RMSE={rmse_train:.3f})")

        if name == "Lasso":

            param = {'alpha': np.arange(0.01, 2, 0.01)}
            best_model, grid = tune_hyperparameters(model, param, X_train, y_train)
            prediction = best_model.predict(X_val)

        if name == "LightGBM":

            #Paramter intervallar henta frå chatgpt med prompt ("Hvilke parametrer og parameter intervaller skal jeg velge for LightGBM")
            param_grid = {
            'num_leaves': [15, 31, 63],
            'max_depth': [-1, 6, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [200, 500, 800],
            'subsample': [0.7, 0.9],
            'colsample_bytree': [0.7, 0.9],
            'reg_lambda': [0, 1, 3]
            }

            best_model, grid = tune_hyperparameters(model, param_grid, X_train, y_train)
            prediction = best_model.predict(X_val)

        if name == "CatBoost":

            #Paramter intervallar henta frå chatgpt med prompt ("Hvilke parametrer og parameter intervaller skal jeg velge for catboost")
            param_grid = {
            'iterations': [200, 500, 800],
            'depth': [6, 8, 10],
            'learning_rate': [0.05, 0.1],
            'l2_leaf_reg': [1, 3, 5],
            'subsample': [0.7, 0.9]
            }
            best_model, grid = tune_hyperparameters(model, param_grid, X_train, y_train)
            prediction = best_model.predict(X_val)

        if name == "XGB":

            #Paramter intervallar henta frå chatgpt med prompt ("Hvilke parametrer og parameter intervaller skal jeg velge for xgboost")
            param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
            }
            best_model, grid = tune_hyperparameters(model, param_grid, X_train, y_train)
            prediction = best_model.predict(X_val)

        rmse_scores[name] = root_mean_squared_error(y_val, prediction)
        models[name] = best_model

        print(f"{name} ferdig – Validering RMSE: {rmse_scores[name]:.3f}")

    best_model_name = min(rmse_scores, key=rmse_scores.get)
    best_model = models[best_model_name]
    best_score = rmse_scores[best_model_name]

    print("\nModelltrening fullført")
    print(f"Beste modell: {best_model_name} (Validering RMSE={best_score:.3f})")

    #Trener beste modell og returnerar den
    train_best_model(best_model, df_test)

    return best_model


# ## Funksjon for å teste beste modell
# 
# **`train_best_model()`** evaluerer den beste modellen med testdatasettet.  
# Følgande steg blir gjennomført:
# 
# - **Datasplitting:** Delar testdatasettet i features (**X_test**) og målvariabel (**y_test**).  
# - **Prediksjon:** Modellen lagar prediksjonar basert på **X_test**.  
# - **Evaluering:** Reknar ut *RMSE* for testdata.  
# 
# Funksjonen skriv ut den endelege *RMSE*-verdien som eit mål på modellens presisjon.
# 

# In[12]:


def train_best_model(best_model, df_test):
    #Delar opp data i variablar X_test og y_test
    X_test = df_test.drop(columns=["free_bikes_next"])
    y_test = df_test["free_bikes_next"]

    prediction = best_model.predict(X_test)
    rmse_score = root_mean_squared_error(y_test, prediction)

    print(f"Den beste modellen hadde en forventet rsme på {rmse_score:.3f}")


# ## Funksjon for å eksportere beste modell
# 
# **`export_best_model()`** lagrar den ferdigtrente beste modellen til ein fil slik at den kan hentast opp og brukast i predict.py.  
# Følgande steg blir gjennomført:
# 
# - **Filnamn:** Opprettar filen **`rental_bikes_predictor.pkl`**.  
# - **Lagring:** Lagrar modellen som pkl.  
# 
# Resultatet er ein **`.pkl`-fil** som kan brukast i predict.py.
# 

# In[13]:


def export_best_model(best_model):

    model_file = "model/rental_bikes_predictor.pkl"

    with open(model_file, "wb") as file:
        pickle.dump(best_model, file)

    print("\nModel successfully exported as pkl")


# ## Main funksjon

# In[14]:


def main():

    df_stations, df_trips, df_weather = load_data()

    df_stations_ready, df_trips_ready, df_weather_ready = prep_data(df_stations, df_trips, df_weather)

    df_model_ready = model_ready(df_stations_ready, df_trips_ready, df_weather_ready)

    #Eksporterer df_model_ready slik at eg kan bruke den til visualisering
    df_model_ready.to_csv("model_ready.csv", index=False)

    print("\nVisualiseringsdata eksportert")

    #Eksporterer model data
    prepped_model_ready = prep_model_ready(df_model_ready)

    df_train, df_val, df_test = train_val_test(prepped_model_ready)

    #Trener modeller og finner beste modell
    best_model = train_models(df_train, df_val, df_test)

    export_best_model(best_model)


# In[15]:


if __name__ == "__main__":
    main()


# In[ ]:




