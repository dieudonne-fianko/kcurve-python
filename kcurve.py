import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import os


_affichage_encodage_fait = False
def charger_et_preparer_donnees(path_csv, colonne_cible):
    """
    Fonction : charger_et_preparer_donnees

    Description :
    Cette fonction permet de charger et préparer des données depuis un fichier CSV, en vue d'une utilisation dans un algorithme de k-NN (k-Nearest Neighbors).
    Les données sont nettoyées, les colonnes inutiles sont supprimées, et les valeurs explicatives sont normalisées.

    Arguments :
    - path_csv (str) : Chemin vers le fichier CSV contenant les données.
    - colonne_cible (str) : Nom de la colonne cible à analyser (variable dépendante).

    Retourne :
    - X (np.ndarray) : Variables explicatives (normalisées).
    - y (np.ndarray) : Variable cible (array 1D encodée numériquement si elle n'est pas déjà numérique).

    Étapes principales :
    1. **Lecture du fichier CSV** :
       - La fonction teste deux séparateurs possibles : ',' et ';'.
       - Vérifie et nettoie les noms de colonnes pour éliminer les espaces et guillemets inutiles.

    2. **Validation de la colonne cible** :
       - Vérifie que la colonne cible existe dans le DataFrame chargé. Si elle est absente, lève une exception avec un message clair.

    3. **Nettoyage des colonnes** :
       - Supprime les colonnes vides (sans valeurs).
       - Supprime les colonnes constantes (sans variance).
       - Supprime les colonnes identifiantes (par exemple : "id" ou "index").

    4. **Encodage de la colonne cible** :
       - Si la colonne cible n'est pas numérique, elle est transformée en catégorie puis encodée numériquement.
       - Affiche une correspondance entre les valeurs originales et les codes numériques (une seule fois grâce à la variable globale `_affichage_encodage_fait`).

    5. **Séparation des variables explicatives et de la cible** :
       - La colonne cible (y) est extraite sous forme d'un tableau 1D.
       - Les variables explicatives (X) sont extraites, avec la colonne cible supprimée.

    6. **Normalisation des variables explicatives** :
       - Utilise StandardScaler pour mettre les données à l'échelle (moyenne 0, variance 1).

    Retour :
    - Retourne X (données explicatives normalisées) et y (colonne cible encodée).

    Remarques :
    - Le fichier CSV doit être correctement formaté et contenir la colonne cible spécifiée.
    - En cas d'erreur (colonne cible absente, mauvais format de fichier), un message explicite est affiché.
    """

    global _affichage_encodage_fait
    try:
        df = pd.read_csv(path_csv, sep=',')
    except:
        df = pd.read_csv(path_csv, sep=';')
    df.columns = df.columns.str.replace('"', '').str.strip()
    if colonne_cible not in df.columns:
        raise ValueError(f"La colonne cible '{colonne_cible}' n'existe pas dans le fichier CSV.")
    df = df.dropna(axis=1, how='all')
    colonnes_uniques = [col for col in df.columns if df[col].nunique() <= 1]
    df = df.drop(columns=colonnes_uniques)
    colonnes_id = [col for col in df.columns if 'id' in col.lower() or 'index' in col.lower()]
    df = df.drop(columns=colonnes_id, errors='ignore')
    if not pd.api.types.is_numeric_dtype(df[colonne_cible]):
        df[colonne_cible] = df[colonne_cible].astype('category')
        if not _affichage_encodage_fait:
            correspondance = dict(enumerate(df[colonne_cible].cat.categories))
            print("\n",f"👉 Correspondance des classes encodées de la variable d'intérêt ({colonne_cible}) :")
            for code, label in correspondance.items():
                print(f"  {code} => {label}")
            _affichage_encodage_fait = True
        df[colonne_cible] = df[colonne_cible].cat.codes
    y = df[colonne_cible].values
    X = df.drop(columns=[colonne_cible]).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y




def calculer_similarites(x_t, X_train):
    """
    Fonction : calculer_similarites

    Description :
    Cette fonction calcule les similarités entre un vecteur donné `x_t` (point cible) et une matrice de données `X_train`
    (ensembles d'entraînement) en utilisant la distance euclidienne. La similarité est ensuite obtenue en appliquant une
    transformation exponentielle inversée sur la distance.

    Arguments :
    - x_t (np.ndarray) : Vecteur cible (1D) dont on souhaite calculer les similarités avec les exemples dans X_train.
    - X_train (np.ndarray) : Matrice (2D) contenant les exemples d'entraînement (chaque ligne représente un exemple).

    Retourne :
    - similarite (np.ndarray) : Tableau contenant les similarités entre le vecteur `x_t` et chaque exemple dans `X_train`.

    Étapes :
    1. Calculer la distance euclidienne entre `x_t` et chaque ligne de `X_train` :
       - La norme euclidienne est utilisée (numpy.linalg.norm avec `axis=1` pour une distance ligne par ligne).
    2. Appliquer une transformation exponentielle inversée sur les distances :
       - Cela permet de convertir des distances (valeurs plus grandes pour des points éloignés) en similarités
         (valeurs plus grandes pour des points proches). La formule utilisée est :
         ```
         similarite = exp(-distance_euclidienne)
         ```.
    3. Retourner le tableau de similarités.

    Applications possibles :
    - Cette fonction peut être utilisée dans des algorithmes de classification (par exemple, k-NN), de clustering, ou d'autres
      techniques nécessitant une mesure de proximité entre points.
    """
    distance_euclidienne = np.linalg.norm(X_train - x_t, axis=1)
    similarite = np.exp(-distance_euclidienne)
    return similarite



def vote_majoritaire(classes_des_voisins):
    """
    Fonction : vote_majoritaire

    Description :
    Cette fonction détermine la classe majoritaire parmi un ensemble de classes fournies (par exemple, les classes des k voisins
    dans un algorithme de k-NN). En d'autres termes, elle retourne la classe la plus fréquemment rencontrée.

    Arguments :
    - classes_des_voisins (list ou np.ndarray) : Une liste ou un tableau contenant les classes (étiquettes) des voisins.

    Retourne :
    - (any) : La classe majoritaire (celle qui apparaît le plus souvent). Si plusieurs classes sont ex æquo, la première rencontrée
      est retournée (par comportement par défaut de `np.argmax`).

    Étapes principales :
    1. Utilisation de `np.unique` :
       - La fonction identifie toutes les classes uniques dans `classes_des_voisins`.
       - Elle compte le nombre d'occurrences de chaque classe.

    2. Détermination de la classe majoritaire :
       - `np.argmax(counts)` trouve l'indice de la classe avec le plus grand nombre d'occurrences.
       - Cet indice est utilisé pour extraire la classe majoritaire depuis `classes`.
    """
    classes, counts = np.unique(classes_des_voisins,return_counts=True)
    return classes[np.argmax(counts)]


def predict(x_t, X_train, y_train, n_neighbors=3):
    """
    Fonction : predict

    Description :
    Cette fonction effectue une prédiction en utilisant l'algorithme k-Nearest Neighbors (k-NN). Elle détermine la classe
    d'un vecteur cible `x_t` en fonction des `n_neighbors` (k voisins les plus proches) dans un ensemble d'entraînement donné.

    Arguments :
    - x_t (np.ndarray) : Vecteur cible (1D) pour lequel une prédiction de classe est effectuée.
    - X_train (np.ndarray) : Matrice des données d'entraînement (2D), chaque ligne représente un exemple d'entraînement.
    - y_train (np.ndarray) : Vecteur des classes (étiquettes) associées à chaque exemple dans `X_train`.
    - n_neighbors (int) : Nombre de voisins les plus proches à considérer (par défaut : 3).

    Retourne :
    - (any) : La classe prédite pour le vecteur cible `x_t`. Elle est déterminée par un vote majoritaire parmi les classes des `n_neighbors`.

    Étapes principales :
    1. **Calcul des similarités :**
       - La fonction `calculer_similarites` est appelée pour calculer les similarités entre `x_t` et chaque exemple dans `X_train`.
       - Les similarités sont converties en distances (plus grande similarité -> plus petit écart).

    2. **Trier les indices par distance croissante :**
       - `np.argsort` trie les indices des exemples d'entraînement en fonction des distances (ordre croissant).
       - L'instruction `indices[::-1]` inverse cet ordre pour obtenir les similarités les plus élevées en premier.

    3. **Sélection des k voisins :**
       - Les `n_neighbors` premiers indices sont sélectionnés.

    4. **Identification des classes des voisins :**
       - À l'aide des indices sélectionnés, les classes associées sont récupérées depuis `y_train`.

    5. **Vote majoritaire :**
       - La classe prédite est déterminée par la fonction `vote_majoritaire`, qui retourne la classe apparaissant le plus fréquemment parmi les voisins.
    """
    indices = np.argsort(calculer_similarites(x_t, X_train))
    kppv = indices[::-1][:n_neighbors]
    classes_des_voisins = y_train[kppv]
    return vote_majoritaire(classes_des_voisins)


def faire_prediction_knn(X_test, X_train, y_train, k=3):
    """
    Fonction : faire_prediction_knn

    Description :
    Cette fonction effectue des prédictions sur un ensemble de données de test (`X_test`) en utilisant l'algorithme k-Nearest Neighbors (k-NN). Pour chaque exemple dans `X_test`, elle utilise les voisins les plus proches dans `X_train` pour déterminer la classe prédite.

    Arguments :
    - X_test (np.ndarray) : Matrice (2D) des exemples à tester (chaque ligne représente un exemple).
    - X_train (np.ndarray) : Matrice (2D) contenant les données d'entraînement (chaque ligne représente un exemple).
    - y_train (np.ndarray) : Vecteur (1D) contenant les classes associées à chaque exemple dans `X_train`.
    - k (int) : Nombre de voisins les plus proches à considérer pour la prédiction (par défaut : 3).

    Retourne :
    - y_pred (np.ndarray) : Vecteur (1D) contenant les classes prédites pour chaque exemple dans `X_test`.

    Étapes principales :
    1. **Initialisation du vecteur des prédictions** :
       - Crée une liste vide `y_pred` pour stocker les classes prédites.

    2. **Boucle sur les exemples de test** :
       - Pour chaque exemple `x_t` dans `X_test` :
         - Appelle la fonction `predict` pour calculer la classe prédite en fonction des `n_neighbors` les plus proches dans `X_train`.
         - Ajoute la prédiction dans la liste `y_pred`.

    3. **Conversion en tableau NumPy** :
       - Convertit la liste `y_pred` en un tableau NumPy pour retourner les prédictions sous forme d'un vecteur.
    """
    y_pred = []
    for x_t in X_test:
        prediction = predict(x_t, X_train, y_train, n_neighbors=k)
        y_pred.append(prediction)
    return np.array(y_pred)



def KNN_error_rate(path_csv, colonne_cible, k=3, test_size=0.3, return_train_error=False):
    """
    Fonction : KNN_error_rate

    Description :
    Cette fonction calcule le taux d'erreur pour un algorithme k-NN (k-Nearest Neighbors) sur un ensemble de test. Elle peut également
    calculer le taux d'erreur sur l'ensemble d'entraînement si l'option `return_train_error` est activée. Le taux d'erreur est défini
    comme le pourcentage de prédictions incorrectes par rapport aux véritables classes.

    Arguments :
    - path_csv (str) : Chemin vers le fichier CSV contenant les données.
    - colonne_cible (str) : Nom de la colonne cible utilisée pour la classification.
    - k (int) : Nombre de voisins à considérer pour la prédiction (par défaut : 3).
    - test_size (float) : Fraction des données utilisée pour l'ensemble de test (par défaut : 0.3, soit 30%).
    - return_train_error (bool) : Indique si le taux d'erreur sur l'ensemble d'entraînement doit être calculé et retourné (par défaut : False).

    Retourne :
    - test_error_rate (float) : Taux d'erreur sur l'ensemble de test (entre 0 et 1).
    - train_error_rate (float, optionnel) : Taux d'erreur sur l'ensemble d'entraînement (si `return_train_error=True`).

    Étapes principales :
    1. **Chargement et préparation des données :**
       - Appelle la fonction `charger_et_preparer_donnees` pour charger les données depuis le fichier CSV et les normaliser.
       - Sépare les données explicatives (X) de la variable cible (y).

    2. **Division en ensembles d'entraînement et de test :**
       - Utilise `train_test_split` pour diviser aléatoirement les données en deux sous-ensembles :
         - `X_train` et `y_train` pour l'entraînement.
         - `X_test` et `y_test` pour le test.

    3. **Prédictions sur l'ensemble de test :**
       - Appelle la fonction `faire_prediction_knn` pour prédire les classes des exemples dans `X_test` en utilisant les données d'entraînement.
       - Calcule le taux d'erreur sur l'ensemble de test comme la proportion de prédictions incorrectes.

    4. **Prédictions sur l'ensemble d'entraînement (optionnel) :**
       - Si `return_train_error=True`, effectue des prédictions sur l'ensemble d'entraînement (`X_train`) et calcule le taux d'erreur correspondant.

    5. **Retour des résultats :**
       - Retourne uniquement le taux d'erreur sur l'ensemble de test, sauf si `return_train_error=True` (dans ce cas, retourne les deux taux).
    """
    X,y = charger_et_preparer_donnees(path_csv,colonne_cible)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    y_pred_test = faire_prediction_knn(X_test, X_train, y_train, k=k)
    test_error_rate = np.mean(y_pred_test != y_test)
    if return_train_error:
        y_pred_train = faire_prediction_knn(X_train, X_train, y_train, k=k)
        train_error_rate = np.mean(y_pred_train != y_train)
        return test_error_rate, train_error_rate
    return test_error_rate


def plot_KNN_error_rate(path_csv, colonne_cible, k_max=10, n_repetitions=10, test_size=0.3):
    """
    Fonction : plot_KNN_error_rate

    Description :
    Cette fonction calcule les taux d'erreur pour un modèle k-NN (k-Nearest Neighbors) en fonction des différentes valeurs de k,
    et trace un graphique présentant les taux d'erreur moyens pour les ensembles de test et d'entraînement. Les intervalles
    de confiance pour chaque valeur de k sont également affichés. De plus, elle génère des fichiers CSV contenant les résultats
    pour chaque répétition, sauvegardés dans un dossier nommé "Tableaux".

    Arguments :
    - path_csv (str) : Chemin vers le fichier CSV contenant les données d'entrée.
    - colonne_cible (str) : Nom de la colonne cible utilisée pour la classification.
    - k_max (int) : Valeur maximale de k à tester (doit être impair, par défaut : 10).
    - n_repetitions (int) : Nombre de répétitions pour chaque valeur de k afin de calculer des moyennes (par défaut : 10).
    - test_size (float) : Fraction des données utilisée pour l'ensemble de test (par défaut : 0.3, soit 30%).

    Étapes principales :
    1. **Initialisation des paramètres :**
       - Génère les valeurs impaires de k à tester (par exemple, k = 1, 3, 5, ... jusqu'à k_max).

    2. **Calcul des taux d'erreur :**
       - Pour chaque répétition, les taux d'erreur pour les ensembles de test et d'entraînement sont calculés à l'aide
         de la fonction `KNN_error_rate`.

    3. **Statistiques des taux d'erreur :**
       - Moyenne des taux d'erreur sur toutes les répétitions pour chaque valeur de k.
       - Intervalles de confiance (5 % et 95 %) calculés pour chaque valeur de k.

    4. **Visualisation des résultats :**
       - Trace les courbes des taux d'erreur moyens pour les ensembles de test et d'entraînement.
       - Affiche les intervalles de confiance pour chaque valeur de k.
       - Identifie la meilleure valeur de k (celle qui minimise le taux d'erreur moyen sur l'ensemble de test).

    5. **Sauvegarde des résultats par répétition :**
       - Crée un dossier "Tableaux" pour sauvegarder les fichiers CSV.
       - Chaque fichier CSV contient les taux d'erreur individuels pour une répétition spécifique.

    Structure des fichiers CSV :
    - Chaque fichier contient deux colonnes :
      - `k` : Les valeurs de k utilisées.
      - `taux_erreur` : Le taux d'erreur pour la répétition correspondante.
    """
    ks = np.arange(1, k_max + 1, 2)
    all_test_error_rates = []
    all_train_error_rates = []
    for n in range(n_repetitions):
        test_error_rates = []
        train_error_rates = []
        for k in ks:
            test_error, train_error = KNN_error_rate(path_csv,colonne_cible, k=k, test_size=test_size, return_train_error=True)
            test_error_rates.append(test_error)
            train_error_rates.append(train_error)
        all_test_error_rates.append(test_error_rates)
        all_train_error_rates.append(train_error_rates)
    all_test_error_rates = np.array(all_test_error_rates)
    all_train_error_rates = np.array(all_train_error_rates)
    mean_test_error = np.mean(all_test_error_rates, axis=0)
    lower_test_bound = np.percentile(all_test_error_rates, 5, axis=0)
    upper_test_bound = np.percentile(all_test_error_rates, 95, axis=0)
    mean_train_error = np.mean(all_train_error_rates, axis=0)
    lower_train_bound = np.percentile(all_train_error_rates, 5, axis=0)
    upper_train_bound = np.percentile(all_train_error_rates, 95, axis=0)
    best_k_index = np.argmin(mean_test_error)
    best_k = ks[best_k_index]
    best_error = mean_test_error[best_k_index]
    plt.figure(figsize=(12, 8))
    plt.fill_between(ks, lower_test_bound, upper_test_bound, color='blue', alpha=0.2, label="Intervalle de confiance (test)")
    sns.lineplot(x=ks, y=mean_test_error, color='blue', linewidth=2, label="Erreur moyenne (test)")
    plt.fill_between(ks, lower_train_bound, upper_train_bound, color='green', alpha=0.2, label="Intervalle de confiance (train)")
    sns.lineplot(x=ks, y=mean_train_error, color='green', linewidth=2, label="Erreur moyenne (train)")
    plt.axvline(x=best_k, color='red', linestyle='--', linewidth=1.5,label=f"Meilleur k = {best_k} (Erreur = {best_error:.2f})")
    plt.title(f"Taux d'erreur du K-NN en fonction de k (avec intervalle de confiance)", fontsize=16)
    plt.xlabel("Valeur de k", fontsize=14)
    plt.ylabel("Taux d'erreur", fontsize=14)
    plt.grid(True)
    plt.xticks(ks)
    plt.legend()
    plt.show()
    dossier = "Tableaux"
    os.makedirs(dossier, exist_ok=True)  # Crée le dossier s'il n'existe pas
    for i in range(n_repetitions):
        df = pd.DataFrame({
            'k': ks,
            'taux_erreur': all_test_error_rates[i]
        })
        nom_fichier = os.path.join(dossier, f'tableau_erreur_repetition_{i+1}.csv')
        df.to_csv(nom_fichier, index=False)
        print("\n",f"📁 Tableau enregistré : {nom_fichier}")




def demander_k_et_calculer_taux_erreur(path_csv, colonne_cible, test_size=0.3):
    """
    Fonction : demander_k_et_calculer_taux_erreur

    Description :
    Cette fonction permet de tester dynamiquement le taux d'erreur d'un algorithme k-NN (k-Nearest Neighbors) pour différentes valeurs
    de k. L'utilisateur peut interagir en ligne de commande pour entrer des valeurs de k à tester. La fonction calcule le taux d'erreur
    associé à chaque valeur de k jusqu'à ce que l'utilisateur décide d'arrêter le test.

    Arguments :
    - path_csv (str) : Chemin vers le fichier CSV contenant les données.
    - colonne_cible (str) : Nom de la colonne cible utilisée pour la classification.
    - test_size (float) : Fraction des données utilisée pour l'ensemble de test (par défaut : 0.3).

    Retourne :
    - Aucun retour explicite, mais affiche le taux d'erreur correspondant à chaque valeur de k entrée par l'utilisateur.

    Étapes principales :
    1. **Chargement et préparation des données :**
       - Appelle la fonction `charger_et_preparer_donnees` pour charger les données depuis le fichier CSV et les normaliser.

    2. **Affichage des instructions :**
       - Explique à l'utilisateur comment entrer des valeurs de k pour tester le taux d'erreur, et comment arrêter le test.

    3. **Boucle d'interaction avec l'utilisateur :**
       - La boucle continue tant que l'utilisateur ne tape pas `'stop'`.
       - Pour chaque entrée utilisateur, vérifie si elle est valide :
         - Si l'entrée est une valeur entière impaire entre 1 et 20, calcule le taux d'erreur avec la fonction `KNN_error_rate`.
         - Si l'entrée est invalide ou une valeur paire, affiche un message d'erreur.

    4. **Calcul et affichage du taux d'erreur :**
       - Calcule le taux d'erreur pour la valeur de k spécifiée et l'affiche avec une précision de 4 chiffres après la virgule.
    """
    charger_et_preparer_donnees(path_csv, colonne_cible)
    print("Test dynamique du taux d'erreur avec différents k")
    print("Entrez un k (impair) pour tester, ou tapez 'stop' pour arrêter.")
    while True:
        k_input = input("Entrez une valeur de k (impair) entre 1 et 20 ou 'stop' pour arrêter : ")
        if k_input.lower() == 'stop':
            print("Arrêt du test.")
            break
        try:
            k = int(k_input)
            if k % 2 == 0:
                print("Veuillez entrer un nombre impair pour k.")
                continue
            error_rate = KNN_error_rate(path_csv,colonne_cible, k=k, test_size=test_size, return_train_error=False)
            print(f"Taux d'erreur moyen pour k={k}: {error_rate:.4f}")
        except ValueError:
            print("Entrée invalide. Veuillez entrer un nombre entier ou 'stop'.")



def meilleur_k(path_csv, colonne_cible, k_max=10, n_repetitions=10, test_size=0.3):
    """
    Fonction : meilleur_k

    Description :
    Cette fonction identifie la meilleure valeur de k (nombre de voisins dans k-NN) en minimisant le taux d'erreur moyen
    sur un ensemble de test. Elle réalise plusieurs répétitions pour chaque valeur de k afin de calculer une moyenne
    fiable des taux d'erreur.

    Arguments :
    - path_csv (str) : Chemin vers le fichier CSV contenant les données.
    - colonne_cible (str) : Nom de la colonne cible utilisée pour la classification.
    - k_max (int) : Valeur maximale de k à tester (doit être impair, par défaut : 10).
    - n_repetitions (int) : Nombre de répétitions pour chaque valeur de k (par défaut : 10).
    - test_size (float) : Fraction des données utilisée pour l'ensemble de test (par défaut : 0.3, soit 30%).

    Retourne :
    - best_k (int) : Meilleure valeur de k identifiée (minimisant le taux d'erreur moyen).
    - best_error (float) : Taux d'erreur moyen minimal pour le meilleur k.
    - best_precision (float) : Précision maximale associée (complément du taux d'erreur : 1 - `best_error`).

    Étapes principales :
    1. **Génération des valeurs impaires de k :**
       - Les valeurs de k à tester sont générées en incrémentant de 2 à partir de 1 jusqu'à `k_max`.

    2. **Initialisation des listes pour stocker les taux d'erreur :**
       - `all_error_rates` conserve les taux d'erreur pour chaque k sur toutes les répétitions.

    3. **Boucle sur les répétitions :**
       - Pour chaque répétition, divise aléatoirement les données en ensembles d'entraînement et de test.
       - Pour chaque valeur de k, calcule le taux d'erreur de test en appelant `KNN_error_rate`.

    4. **Calcul des statistiques :**
       - Convertit les taux d'erreur en un tableau NumPy pour faciliter le calcul des moyennes.
       - Détermine la moyenne des taux d'erreur pour chaque valeur de k.
       - Trouve l'indice correspondant au plus petit taux d'erreur moyen (meilleur k).

    5. **Calcul des métriques associées au meilleur k :**
       - Taux d'erreur minimal (`best_error`) et précision maximale (`best_precision`).

    6. **Affichage et retour des résultats :**
       - Affiche le meilleur k trouvé, le taux d'erreur minimal et la précision maximale.
       - Retourne ces valeurs pour une utilisation ultérieure.
    """
    ks = np.arange(1, k_max + 1, 2)
    all_error_rates = []
    for n in range(n_repetitions):
        error_rates = []
        for k in ks:
            error_rate= KNN_error_rate(path_csv, colonne_cible, k=k, test_size=test_size,return_train_error=False)
            error_rates.append(error_rate)
        all_error_rates.append(error_rates)
    all_error_rates = np.array(all_error_rates)
    mean_error_rate = np.mean(all_error_rates, axis=0)
    best_k_index = np.argmin(mean_error_rate)
    best_k = ks[best_k_index]
    best_error = mean_error_rate[best_k_index]
    best_precision = 1 - best_error
    print("\n",f"Le meilleur k est : {best_k}")
    print("\n",f"Taux d'erreur moyen minimal : {best_error:.4f}")
    print("\n",f"Précision maximale : {best_precision:.4f}")
    return best_k, best_error, best_precision



def afficher_voisins(path_csv, colonne_cible, k_max, n_repetitions, test_size):
    """
    Fonction : afficher_voisins

    Description :
    Cette fonction permet à l'utilisateur de sélectionner un patient (échantillon) depuis un ensemble de test et
    de visualiser ses k voisins les plus proches en utilisant l'algorithme k-NN. Une réduction en 2 dimensions
    (PCA) est utilisée pour représenter les données sur un graphique.

    Arguments :
    - path_csv (str) : Chemin vers le fichier CSV contenant les données.
    - colonne_cible (str) : Nom de la colonne cible à prédire (exemple : 'diagnosis').
    - k_max (int) : Valeur maximale de k à tester pour déterminer le meilleur k.
    - n_repetitions (int) : Nombre de répétitions pour optimiser le choix du meilleur k.
    - test_size (float) : Fraction des données réservée à l'ensemble de test (par défaut : 0.3, soit 30 %).

    Étapes principales :
    1. **Chargement et préparation des données :**
       - Charge les données depuis le fichier CSV et sépare les variables explicatives (X) et la cible (y).
       - Divise les données en ensembles d'entraînement et de test (X_train, X_test, y_train, y_test).

    2. **Détermination du meilleur k :**
       - Identifie la valeur optimale de k (best_k) basée sur le taux d'erreur moyen à l'aide de la fonction `meilleur_k`.

    3. **Interaction utilisateur :**
       - Demande à l'utilisateur d'entrer un index correspondant à un patient dans l'ensemble de test.
       - L'utilisateur peut saisir un numéro de patient ou taper `'stop'` pour quitter.

    4. **Recherche des voisins les plus proches :**
       - Utilise `NearestNeighbors` pour trouver les `best_k` voisins les plus proches du patient sélectionné.
       - Exclut le patient lui-même des voisins.

    5. **Réduction de dimension avec PCA :**
       - Réduit les données en 2 dimensions pour permettre une visualisation graphique.

    6. **Affichage graphique :**
       - Affiche le patient sélectionné (en bleu) ainsi que ses voisins les plus proches (en rouge) sur une carte 2D.
       - L'ensemble d'entraînement est également affiché (en gris clair) pour fournir un contexte.
    """
    X,y = charger_et_preparer_donnees(path_csv, colonne_cible)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    best_k, best_error, best_precision = meilleur_k(path_csv, colonne_cible, k_max, n_repetitions, test_size)
    while True:
        saisie = input(f"🧪 Entrez le numéro d’un patient (0 à {len(X_test)-1}) ou tapez 'stop' pour quitter : ")
        if saisie.lower() == 'stop':
            print("✅ Fin de l'affichage.")
            break
        try:
            index = int(saisie)
            if index < 0 or index >= len(X_test):
                print("❌ Index invalide. Essayez encore.")
                continue
        except ValueError:
            print("❌ Entrée invalide. Veuillez entrer un nombre ou 'stop'.")
            continue
        x_test = X_test[index]
        y_test_label = y_test[index]

        # Calcule des similarités en utilisant NearestNeighbors avec le meilleur k
        knn = NearestNeighbors(n_neighbors=best_k + 1)  # +1 pour inclure le patient lui-même
        knn.fit(X_train)  # Entraînement du modèle sur les données d'entraînement

        # Trouver les indices des k voisins les plus proches du patient sélectionné
        distances, indices_les_plus_proches = knn.kneighbors([x_test])

        # Exclure le patient lui-même (le premier voisin dans la liste)
        indices_les_plus_proches = indices_les_plus_proches[0][1:]  # Enlever l'index du patient lui-même

        # Réduction en 2D pour affichage (PCA)
        pca = PCA(n_components=2)
        X_train_2D = pca.fit_transform(X_train)
        x_test_2D = pca.transform([x_test])[0]
        voisins_2D = X_train_2D[indices_les_plus_proches]

        # Affichage du graphique
        plt.figure(figsize=(8, 6))
        plt.scatter(X_train_2D[:, 0], X_train_2D[:, 1], c='lightgray', label='Données entraînement', alpha=0.5)
        plt.scatter(voisins_2D[:, 0], voisins_2D[:, 1], c='red', label=f'{best_k} voisins les plus proches', s=80)
        plt.scatter(x_test_2D[0], x_test_2D[1], c='blue', label=f'Patient N°{index}', marker='X', s=150)

        # Zoom automatique
        all_x = np.concatenate([voisins_2D[:, 0], [x_test_2D[0]]])
        all_y = np.concatenate([voisins_2D[:, 1], [x_test_2D[1]]])
        plt.xlim(all_x.min() - 1, all_x.max() + 1)
        plt.ylim(all_y.min() - 1, all_y.max() + 1)

        plt.title(f"Patient N°{index} et ses {best_k} voisins les plus proches")
        plt.legend()
        plt.grid(True)
        plt.show()


# Protocole de test en ligne de commande
def main():
    """
    Cette fonction `main()` permet d'exécuter différents tests liés au k-NN (k-Nearest Neighbors) à partir de la ligne de commande.
    Elle utilise argparse pour gérer les paramètres fournis par l'utilisateur et permet de configurer les analyses de manière dynamique.

    ### Utilisation :
    Depuis le terminal, vous pouvez lancer ce script en spécifiant les arguments suivants :

    - `--data` : Chemin vers le fichier CSV contenant les données (obligatoire).
    - `--colonne_cible` : Nom de la colonne cible à analyser pour la classification (obligatoire).
    - `--k_max` : Valeur maximale de k à tester pour les courbes d'erreur (optionnel, défaut : 10).
    - `--k` : Valeur spécifique de k à tester pour le taux d'erreur (optionnel, défaut : 3).
    - `--n_repetitions` : Nombre de répétitions pour obtenir des résultats fiables (optionnel, défaut : 10).
    - `--test_size` : Fraction des données réservée au test (optionnel, défaut : 0.3).
    - `--action` : Action à effectuer (obligatoire), parmi :
        - `charger_et_preparer_donnees` :  Charger et préparer un dataframe (but : séparer la variable dépendante des variables explicatives.
        - `KNN_error_rate` : Calculer le taux d'erreur pour une valeur spécifique de k.
        - `plot_KNN_error_rate` : Tracer les courbes de taux d'erreur en fonction des valeurs de k.
        - `k_rate_intuitive` : Calculer étape par étape le taux d'erreur de manière intuitive.
        - `meilleur_k` : Identifier la valeur optimale de k en minimisant le taux d'erreur.
        - `afficher_voisins` : Afficher un graphique montrant un patient avec ses k plus proches voisins ainsi que les point d'entrainement.
    """
    parser = argparse.ArgumentParser(description="Protocole de test pour k-NN avec k-curves")
    parser.add_argument("--data", type=str, required=True,
                        help="Chemin vers le fichier CSV contenant les données.")
    parser.add_argument("--colonne_cible", type=str, required=True,
                        help="Nom de la colonne cible à analyser.")
    parser.add_argument("--k_max", type=int, default=10,
                        help="Valeur maximale de k à tester (doit être impair).")
    parser.add_argument("--k", type=int, default=3,
                        help="Valeur de k à tester (doit être impair).")
    parser.add_argument("--n_repetitions", type=int, default=10,
                        help="Nombre de répétitions pour chaque valeur de k.")
    parser.add_argument("--test_size", type=float, default=0.3,
                        help="Proportion des données utilisées pour le test.")
    parser.add_argument("--action", type=str, choices=["charger_et_preparer_donnees","KNN_error_rate","plot_KNN_error_rate","k_rate_intuitive","meilleur_k","afficher_voisins"],
                        required=True, help="Action à effectuer ("
                                            "'charger_et_preparer_donnees' pour charger et préparer un dataframe (but : séparer la variable dépendante des variables explicatives,"
                                            "'KNN_error_rate' pour calculer le taux d'erreur en fonction du k, "
                                            "'plot_KNN_error_rate' pour tracer les courbes de taux t'erreur en fonction des valeurs de K allant de 1 à kmax,"
                                            "'k_rate_intuitive'pour aller étape par étape de manière intuitive dans le calcul du taux d'erreur en fonction du k,"
                                            "'meilleur_k' pour trouver le meilleur k),'afficher_voisins' pour afficher un graphique montrant un patient avec ses k plus proches voisins ainsi que les point d'entrainement")
    args = parser.parse_args()
    try:
        charger_et_preparer_donnees(args.data, args.colonne_cible)
    except Exception as e:
        print(f"Erreur lors du chargement des données : {e}")
        return
    if args.action == "charger_et_preparer_donnees" :
        X,y =charger_et_preparer_donnees(
            path_csv=args.data,
            colonne_cible=args.colonne_cible
        )
        print("\n","Liste des variables explicatives","\n",X)
        print("\n","Liste de la variable d'intérêt","\n",y)
    elif args.action == "KNN_error_rate" :
        error_rate=KNN_error_rate(
            path_csv=args.data,
            colonne_cible=args.colonne_cible,
            k=args.k,
            test_size=args.test_size,
            return_train_error=False
        )
        print("\n",f"Taux d'erreur pour k={args.k} : {error_rate:.4f}")
    elif args.action == "plot_KNN_error_rate":
        plot_KNN_error_rate(
            path_csv=args.data,
            colonne_cible=args.colonne_cible,
            k_max=args.k_max,
            n_repetitions=args.n_repetitions,
            test_size=args.test_size
        )
    elif args.action == "k_rate_intuitive" :
        demander_k_et_calculer_taux_erreur (
            path_csv=args.data,
            colonne_cible=args.colonne_cible,
            test_size=args.test_size
        )
    elif args.action == "meilleur_k":
        best_k, best_error, best_precision = meilleur_k(
            path_csv=args.data,
            colonne_cible=args.colonne_cible,
            k_max=args.k_max,
            n_repetitions=args.n_repetitions,
            test_size=args.test_size
        )
        print("\n",f"✅ Meilleur k trouvé : {best_k}")
        print("\n",f"Taux d'erreur minimal : {best_error:.4f}")
        print("\n",f"Précision maximale associée : {best_precision:.4f}")
    elif args.action == "afficher_voisins" :
        afficher_voisins(
            path_csv=args.data,
            colonne_cible=args.colonne_cible,
            k_max=args.k_max,
            n_repetitions=args.n_repetitions,
            test_size=args.test_size
        )

if __name__ == "__main__":
    main()

