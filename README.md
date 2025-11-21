# Før du køyrer programmet må du ha følgande biblioteker
- **numpy**
- **pandas**
- **matplotlib**
- **seaborn**
- **scikit-learn**
- **xgboost**
- **lightgbm**
- **catboost**

Dersom desse manglar kan instasller de slik: pip instsall catboost xgboost pandas osv..

## 1. Pakkut ZIP filen
## 2. Naviger til mappen i terminal (cmd, anaconda, powershell)
## 3. Køyr main med: python main.py

# Om oppgåva
Dette Data Science prosjektet fekk ein score på 91/100 poeng. Trekk av poeng kom frå data lekkasje somme plassar pga gjennomsnittsdata fra heile
datasettet vart brukt for å fylle tomme verdiar. Her skulle bare gjennomsnittsverdiar frå det gjeldande datasettet bli brukt for å gjere dette. F.eks
eksisterar det manglande verdiar i trains.csv skal data kun fra det datasettet bli brukt til å fylle inn de manglande verdiane.

Prosjektet består av tre program: main, train og predict.
- **train** Rydder og slår saman data samt trenar opp og finner beste model
- **predict:** Bruker den utvalte modellen til å predikere ledige syklar ved gitte stasjoner ein time fram i tid
- **main** Køyrer begge program, først train, så predict

<img width="571" height="579" alt="image" src="https://github.com/user-attachments/assets/3dfd302b-7ad1-46c8-9ae0-216e101ad929" />

<img width="709" height="256" alt="image" src="https://github.com/user-attachments/assets/2dd945db-cb2c-4286-90c6-b40a067db32f" />
