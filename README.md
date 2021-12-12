# Rain in Australia
**Alumne** : Ricard Lopez Olivares (ricardlol en github)<br />
**Dataset** : [Rain in Australia link](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package)

# Taula de continguts

1. [Objectiu del repositori](#Objectiu-del-repositori)
2. [Estructura del repositori](#Estructura-del-repositori)
3. [Objectiu del dataset aplicat](#Objectiu-del-dataset-aplicat)
4. [Models aplicats](#Models-aplicats)
5. [Comparativa de models](#Comparativa-de-models)
6. [Comparativa amb altres kaggles](#Comparativa-amb-altres-kaggles)

## Objectiu del repositori

En el repositori podem trobar un seguit de mètodes que es poden utilitzar per resoldre problemes d'aprenentatge computacional. El codi té dues versions, la primera està en un fitxer .py, la segona és en Jupyter Notebook, la qual aquesta esta pensada perquè sigui més interactiu i mostrar amb més claredat el contingut.

## Estructura del repositori

El repositori està dividit en diverses seccions:
- Data : Repositori on trobarem tots els datasets utilitzats
- Demo : Demostracions petites per mostrar apartats en concret, com per exemple widgets implementats, Oversampling, etc
- Figures : Plots, imatges utilitzades per mostrar visualment amb el que es treballa
- Notebooks: Directori on trobarem els notebooks de jupyter
    - Archive : notebook en producció, és un notebook on poden trobar errades.
    - Develop : notebook funcional sense errors.

## Objectiu del dataset aplicat

Com podem preveure sobre el títol del dataset, el que volem predir, si amb una sèrie d'atributs, com la localització, els núvols, etc., si plourà demà o no.

## Models aplicats

Es podrien aplicar qualsevol model de classificació, en aquest repositori trobarem el següents:
- [Regressió logística](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [Decision tree](https://scikit-learn.org/stable/modules/tree.html)
- [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Xgboost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
- [KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

## Comparativa de models
### Models per defecte amb dades estandaritzades i oversampling

| Model | Hiperparametre | Accuracy | Recall | Precision| F1-score | Temps (s)|
| ----- | ----| ---- |---- |---- |---- |---- |
| Regressió logística | No Parameters | 0.789| 0.763 | 0.795 |0.789| 2.89|
| SVC |  No Parameters | - | - | - | - | no acaba |
| Decision Tree | max_depth = 6 | 0.763| 0.739 | 0.772 | 0.759 | 3.174|
| Random Forest | max_samples = 0.9 | 0.946| 0.975 | 0.92 | 0.946| 406.40|
| XGBOOST | objective='binary:logistic' <br /> random_state=0 | 0.837 | 0.838 | 0.837 | 0.836 | 15.85|
| KNN |  n_neighbors=5 | 0.802| 0.838 | 0.837| 0.801| 0.263|

### Models sense estandaritzar i oversampling

| Model | Hiperparametre | Accuracy | Recall | Precision| F1-score | Temps (s)|
| ----- | ----| ---- |---- |---- |---- |---- |
| Regressió logística | No Parameters | 0.777| 0.759 | 0.781 |0.774| 3.14|
| SVC |  No Parameters | - | - | - | - | no acaba |
| Decision Tree | max_depth = 6 | 0.766| 0.732 | 0.787 |0.762| 2.65|
| Random Forest | max_samples = 0.9 | 0.949| 0.977 | 0.919 |0.949| 406.40|
| XGBOOST | objective='binary:logistic' <br /> random_state=0 | 0.823 | |  |0.823 | 25.70|
| KNN |  n_neighbors=5 | 0.802| |  | 0.801| 0.263|

### Models utilitzant skew amb oversampling

| Model | Hiperparametre | Accuracy | Recall | Precision| F1-score | Temps (s)|
| ----- | ----| ---- |---- |---- |---- |---- |
| Regressió logística | No Parameterse | 0.761| |  |0.761| 5.62|
| Decision Tree | max_depth = 6 | 0.767| |  |0.767| 3.41|
| Random Forest | max_samples = 0.9 | 0.945| 0.977 | 0.919 |0.945| 52.40|
| XGBOOST | objective='binary:logistic' <br /> random_state=0 | 0.838 | 0.837 | 0.836 | 0.838 | 34.42|
| KNN |  n_neighbors=5 | 0.817| 0.888 | 0.791 | 0.817| 0.232|

### Models utilitzant skew y standarització amb oversampling

| Model | Hiperparametre | Accuracy | Recall | Precision| F1-score | Temps (s)|
| ----- | ----| ---- |---- |---- |---- |---- |
| Regressió logística | No Parameters | 0.787| 0.774 | 0.794 | 0.787| 5.70|
| Decision Tree | max_depth = 6 | 0.767| 0.775 | 0.763 | 0.767| 3.41|
| Random Forest | max_samples = 0.9 | 0.945| 0.977 | 0.919 | 0.945| 52.40|
| XGBOOST | objective='binary:logistic' <br /> random_state=0 | 0.838 | 0.838 | 0.835 | 0.838 | 34.42|
| KNN |  n_neighbors=5 | 0.817| 0.861 |  0.776 | 0.888 | 0.232|

Com podem veure, utilitzar un skew no ens donà cap benefici en scores, però si en temps, sobretot en els models que trigaven més com el random forest.

A partir mirare de millorar el millor model que tenim modificant els hiperparametres.

### Random forest canviant hiperparametres

El millors hiperparametres per el random forest son els següents:
- n_estimators= 50
- max_samples= 0.3
- max_features='log2'
- criterion= 'gini'
- bootstrap= False

Aquest hiperparametres hs sigut vtinguts amb un RandomizedSeardfchCV.
Amb aquest parametres ens donara una accuracy de 0.96, un F1_score de 0.96 a,b un temps de 30.11 segons.

## Comparativa amb altres kaggles

Si mirem repositoris d'altres persones, podem veure títols amb un acuracy molt alta, però observant el contingut podem veure, el següent:
- No balanceja'n el target, per tant, poden obtenir un model que predigui bé amb els valors de la majoria, però realment aquest model no prediu bé realment, ja que les prediccions de la classe minoritària els farà bastant malament la majoria de vegades, un exemple d'això ho veurem en el kaggle de [simonbeylat](https://www.kaggle.com/simonbeylat/nn-with-keras-99-accuracy), el qual no tracta la variable objectiu.

- Uns altres el que fan és eliminar outlayers, aquesta decisió ens ara tenir menys dades, però com tenim moltes, no passa res. Un exemple d'això ho tenim en aquest mateix kaggle o si escau un exemple extern tenim el kaggle de [Muhammad Shahbaz Muneer](https://www.kaggle.com/mdshahbazmuneer/91-accuracy-complete-explanation-with-comments).

- Per últim tenim kaggles que dona un acuracy més baix aquest és perquè no tracten les dades, es a dir, no normalitzen o estandarditzen, per tant, les dades estan bastant dispersés, un exemple d'això el tenim amb el kaggle de [lynnxy](https://www.kaggle.com/lynnxy/rain-in-australia-eda-ml).



