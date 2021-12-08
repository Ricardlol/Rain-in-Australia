# Rain in Australia
**Alumne** : Ricard Lopez Olivares (ricardlol en github)__
**Dataset** : [Rain in Australia link](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package)

# taula de continguts

1. [Objectiu del repositori](#Objectiu-del-repositori)
2. [Estructura del repositori](#Estructura-del-repositori)
3. [Objectiu del dataset aplicat](#Objectiu-del-dataset-aplicat)
4. [Models aplicats](#Models-aplicats)

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
