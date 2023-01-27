# TP GPGPU :  réseau de neurones

Dans ce TP, nous avons un code séquentiel écrit en C++ qui permet d'entraîner un réseau de neurones. Nous avons parallélisé ce code pour GPU en utilisant CUDA afin de diminuer le temps de calcul.

Vous trouverez dans ce git :
- code_TP_original : ce fichier possède le code de base en C++
- code_TP : ce dossier possède les différents fichiers qui parallélisent le code en CUDA et un dossier avec deux fichiers d'analyse du temps de calcul sur CPU avec et sans GPU
- img_train : images d'entaînement
- img_test : images de test

Pour compiler le code en CUDA : nvcc -o ./ann main.cu matrix.cu ann.cu mnist.cu -lm  
Pour run le code : ./ann
