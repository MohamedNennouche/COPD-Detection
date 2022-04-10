# COPD Detection
## But du projet 
Ce projet est un problème de classification se basant sur un dataset dans le domaine de l'Electronic Nose pour la détection de la maladie pulmonaire obstructive chronique (COPD) parmi des patients sains et d'autres patients fumeurs en se basant sur les résultats d'un réseau de capteurs de gaz (8 capteurs), les données et les détails des données et la méthodologie de mesures sont décrites suivant ce [lien](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7838708/?fbclid=IwAR2os2eFQn2hjPFu84r4nB5TY7bxbdmkbwg9kzGoeXLuNeOuU5ExJpcw6rw). 
## Structure du dépôt 
Le projet se constitue de plusieurs dossiers et fichiers décrits comme suit : 
### Les fichiers
On a principalement des notebook Jupyter et un fichier python : 
- **COPD-notebook.ipynb :** Ce fichier sert à l'exploration des données d'origine et nous permettre de mettre en comparaison les graphiques des différents capteurs pour nos 4 classes et l'enregistrement sous forme d'images et de tableau Numpy pour des expérimentations futures. 
- **COPD-cnn-classifier.ipynb :** Ce fichier englobe une classification utilisant un CNN utilisant les images enregistrées avec le fichier précédent. 
- **COPD-one-1D-signal.ipynb :** Ce fichier contient plusieurs expérimentations sur de la classification utilisant les signaux directement, en utilisant un KNN, un SVM et ensuite un réseau de neurones CNN appliqué sur chaque signal de capteur indépendemment
- **COPD-all-1D-signal.ipynb :** Ce fichier contient une préparation des données capteurs (les 8 capteurs) et les entrer directement à un CNN en empilant les datasets, ayant de très bons résultats. 
- **COPD-deepInsight :** Ce fichier contient une méthode permettant de transformer des données non images en images qui s'appelle **DeepInsight**, dont la documentation se trouve à ce [lien](https://github.com/alok-ai-lab/pyDeepInsight) permettant leurs utilisations en entrée de CNN comme des images sur des problèmes de détections ou de classifications. 
- **COPD-all-1D-augmentation :** Ce fichier développe plusieurs méthodes d'augmentation des données essayant d'améliorer les résultats sur les méthodes avancées précédemment. 
### Les dossiers