# COPD Detection
## But du projet 
Ce projet est un problème de classification se basant sur un dataset dans le domaine de l'Electronic Nose pour la détection de la maladie pulmonaire obstructive chronique (COPD) parmi des patients sains et d'autres patients fumeurs en se basant sur les résultats d'un réseau de capteurs de gaz (8 capteurs), les données et les détails des données et la méthodologie de mesures sont décrites suivant ce [lien](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7838708/?fbclid=IwAR2os2eFQn2hjPFu84r4nB5TY7bxbdmkbwg9kzGoeXLuNeOuU5ExJpcw6rw). 
## Structure du dépôt 
Le projet se constitue de plusieurs dossiers et fichiers décrits comme suit : 
### Les fichiers
On a principalement des notebook Jupyter et un fichier python : 
- **COPD-notebook.ipynb :** Ce fichier sert à l'exploration des données d'origine et nous permettre de mettre en comparaison les graphiques des différents capteurs pour nos 4 classes et l'enregistrement sous forme d'images et de tableau Numpy pour des expérimentations futures. 
- **COPD-cnn-classifier.ipynb :** Ce fichier englobe une classification utilisant un CNN utilisant les images enregistrées avec le fichier précédent. 
- **COPD-one-1D-signal.ipynb :** Ce fichier contient plusieurs expérimentations sur de la classification utilisant les signaux directement, en utilisant un KNN, un SVM et ensuite un réseau de neurones CNN appliqué sur chaque signal de capteur indépendemment.
- **COPD-all-1D-signal.ipynb :** Ce fichier contient une préparation des données capteurs (les 8 capteurs) et les entrer directement à un CNN en empilant les datasets, ayant de très bons résultats. 
- **COPD-deepInsight.ipynb :** Ce fichier contient une méthode permettant de transformer des données non images en images qui s'appelle **DeepInsight**, dont la documentation se trouve à ce [lien](https://github.com/alok-ai-lab/pyDeepInsight) permettant leurs utilisations en entrée de CNN comme des images sur des problèmes de détections ou de classifications. 
- **COPD-all-1D-augmentation.ipynb :** Ce fichier développe plusieurs méthodes d'augmentation des données essayant d'améliorer les résultats sur les méthodes avancées précédemment. 
- **utilities.py :** C'est un fichier contenant les fonctions utilisées dans les autres notebook et sert alors de librairie de fonctions. 
### Les dossiers
#### Dépendances
- **__pycache__ :** Permettant de faire le lien entre les fonctions présentent dans le fichier utilities et les notebooks où on les utilise. 
#### Données
- **text_files :** C'est les fichiers originaux (qu'on trouve dans le lien du datasets), chaque fichier représente une classe. 
- **csv_files :** C'est les fichiers équivalents sous format csv.
- **images :** C'est les images générées en restructurant les signaux dans le fichier COPD-notebook, chaque dossier contient les images se sa classe. 
- **Training, Validation, Test :** C'est la distribution des images précédentes pour l'entrainement, la validation et le test du modèle. 
- **numpy_arrays :** C'est les mêmes matrices que les images sans encodage sur 8 bits, en gardant les valeurs flottantes originales. 
- **capteurs :** C'est les données de chaque capteurs pris à part entière en précisant la classe de patient que le capteur mesure. 
#### Modèles
Tous les autres dossiers c'est les modèles de deep learning enregistrés durant leurs entraînements dans les différents notebook. 
## Performances atteintes