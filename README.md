# Projet 8 - Formation IA Engineer d'OpenClassrooms

## Traitez les images pour le système embarqué d’une voiture autonome

### Contexte

Vous êtes ingénieur IA chez "Future Vision Transport", une entreprise qui conçoit des systèmes embarqués de vision par ordinateur pour les véhicules autonomes.

Notre mission est de concevoir un premier modèle de segmentation d’images basé sur le framework Keras.

Ce modèle devra être déployé via une API FastAPI sur le Cloud Azure pour qu'il soit utilisé par les collègues du système de décision.

Cette API prendra en entrée une image et renvoie l'image, le masque réel et le masque préditpar l'API.

### Notebooks complets et commentés ci-dessous :

https://github.com/CelineLesur/OCR_IAEngineer_Projet8/blob/3e2d224de5ef23420196904d508d20d5a4c07417/notebooks/P8_EDA.ipynb

https://github.com/CelineLesur/OCR_IAEngineer_Projet8/blob/3e2d224de5ef23420196904d508d20d5a4c07417/notebooks/P8_DataPreprocessing.ipynb

### Découpage des dossiers :
📂 /

main.py → Code principal de l’API FastAPI

requirements.txt → Liste des packages nécessaires

README.md → Explication du contexte du projet, de la hierarchie des fichiers et des packages utilisés

📂 notebooks/

P8_EDA.ipynb → Analyse exploratoire des données

P8_Datapreprocessig.ipynb → Entraînement et évaluation du modèle de segmentation de type U-net

DataGenerator.py → Script permettant  de faire des batchs d'images et de leur appliquer le pré-traitement

### Installation

#### Prerequisites

Python 3.9

#### Dependencies

- fastapi==0.115.1
- numpy==1.24.4
- pillow
- tensorflow-cpu==2.10.0
- keras==2.10.0
- git+https://github.com/qubvel/segmentation_models.git
- azure-storage-blob
- matplotlib
- opencv-python-headless==4.5.5.64
- python-multipart
- uvicorn[standard]
