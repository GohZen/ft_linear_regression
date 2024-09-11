import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Calcule le temps de process a chaque étape (calcul le cout ou l'erreur moyenne quadratique du modèle de régression)
# - predicted_price = liste de val prédites par le modèle, basé sur theta0 et theta1 ainsi que les data d'entrée (kilométrage)
# - price = liste des prix réels correspondant aux données d'entrainement.
# - m = nombre total de points de données dans le dataset (nbr d'observation dispo)
#
# Pour chaque points de donnés, on calcul le carré de la différence entre la valeur prédite par le modèle (predicted_price[i])
# et la valeur réelle (price[i]), cela permet de mesurer l'erreur quadratique pour ce point en particulier.
#
# La somme de toutes les erreurs quadratique ainsi récupérées est effectuée par la suite afin de la normaliser via '(1/(2*m))',
# c est a dire en divisant la somme par "2*m", ce qui permet d'obtenir une mesure d'erreur moyenne quadratique.
def compute_cost(predicted_price, price):  
    m = len(price)
    return (1/(2*m)) * np.sum([(price[i] - predicted_price[i]) ** 2 for i in range(m)])

# Charge les données comprise dans le fichier CSV
# Utilise la librairie Pandas pour lire le CSV et charger les données dans un dataframe nommé 'data'. il s'agit d'une structure
# de donnés bidimensionnelle facilitant la manipulation et l'analyse des donnés tabulaires.
# La fonction récupèrent 2 colonnes spécifiques du dataframe : 
# - km_count = contient les valeurs des kilométrages pour chaque voitures, extraint de la colonne 'km' 
# - price = contient les prix des voitures, extrait de la colonne 'price'
# Renvoie 2 tableau (km_count et price)
def load_data(file_path='ressources/data.csv'):
    data = pd.read_csv(file_path)
    km_count = data['km'].values
    price = data['price'].values

    print("Km Count - Min:", km_count.min(), "Max:", km_count.max(), "Moyenne:", km_count.mean())
    print("Price - Min:", price.min(), "Max:", price.max(), "Moyenne:", price.mean())

    return km_count, price

# Utilisée pour normaliser les données. Il s'agit d'une étape importante de le prétraitement des datas.
# - data = tableau des données a normaliser
# - min_val / max_val = valeurs minimal / maximal trouvé dans le tableau de donnés
# Les valeurs min / max sont utilisées pour mettre les donnés a l'échelle. La normalisation conciste a transformer
# les valeurs d'une plage d'origine en une plage différente (souvent entre 0 et 1), afin d'améliorer les performances 
#
# "[(x - min_val) / (max_val - min_val) for x in data]"
# --> Pour chaque valeurs 'x' dans 'data', on calcul une valeur normalisée en soustrayant la valeur minimale et en 
#     divisant par l'étendue des valeurs (max_val - min_val). Cela met à l'échelle toutes les valeurs pour qu'elles 
#     soient dans l'intervalle [0,1].
# 
# La fonction retourne 3 éléments : 
# - Une liste de donnés normalisés
# - La valeur minimale originale (min_val)
# - La valeur maximale originale (max_val)
def normalize(data):
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) for x in data], min_val, max_val

# Utilisée pour convertir des valeurs normalisées (c'est-à-dire mises à l'échelle) de retour dans leur plage d'origine.
# Prend 3 éléments en paramètre : 
# - value = valeur normalisée que l'on souhaite convertir de retour dans la plage d'origine
# - min_val = valeur minimale d'origine (avant normalisation)
# - max_val = valeur maximale d'origine (avant normalisation)
# 
# Formule de dénormalisation : "value * (max_val - min_val) + min_val"
# Cette formule inverse le processus de normalisation. 
# Sachant que : normalisation = (ORIGINAL_VAL - MIN_VAL) / (MAX_VAL - MIN_VAL)
# Alors :       dénormalisation = NORMALIZED_VAL * (MAX_VAL - MIN_VAL) + MIN_VAL
#
# La multiplication par (MAX_VAL - MIN_VAL) ajuste la plage de valeurs normalisée pour qu'elle corresponde a l'échelle d'origine.
# L'ajout de MIN_VAL ajuste la valeur pour qu'elle corresponde a la valeur minimale d'origine
def denormalize(value, min_val, max_val):
    return value * (max_val - min_val) + min_val

# Applique la descente de gradient pour optimiser theta0 et theta1
# - km_count       =  Les valeurs de kilométrage (caractéristique du modèle)
# - price          =  Les valeurs cibles (prix réels des voitures)
# - theta0, theta1 =  Parametre du modèle de regression que nous allons ajuster (init a 0)
# - alpha          =  Taux d'apprentissage (taille des pas) dans la mise a jour des paramètre
# - iter_count     =  Nombre d'itération pour exécuter la descente de gradient
# - m              =  Nombre d'exemple dans les donnés (nbr d'observations)
#
# 'sum_error_theta0' et 'sum_error_theta1' sont initialisés pour accumuler les gradients pour 'theta0' et 'theta1' respectivement.
# 
# DESCENTE DE GRADIENTS
# - Boucle sur les exemples
# - 'predicted_price' donne la valeur prédite par le modèle pour l'exemple 'j'
# - 'error' done la différence entre la valeur prédite et la valeur réelleµ
# - Les gradients 'sum_error_theta0' et 'sum_error_theta1' mis à jour en ajoutant respectivement 'error' et 'error * km_count[j]'.
# 
# CALCUL DES GRADIENTS MOYENS
# - 'd_theta0' et 'd_theta1' sont les moyennes des gradients accumulés pour theta0 et theta1. 
# Cela donne la direction et la magnitude du changement nécessaire pour minimiser l'erreur.
#
# MISE A JOUR DES PARANETRES
# Les paramètres 'theta0' et 'theta1' sont ajustés en soustrayant les gradients multipliés par le taux d'apprentissage alpha. 
# Cela ajuste les paramètres dans la direction qui minimise l'erreur.
# 
# Après avoir complèté les itérations, les paramètres optimisés 'theta0' et 'theta1' sont retournés pour être utilisés dans les prédictions.
def do_gradient_descent(km_count, price, theta0, theta1, alpha, iter_count):
    m = len(km_count)
    for i in range(iter_count):
        sum_error_theta0 = 0
        sum_error_theta1 = 0
        
        for j in range(m):
            predicted_price = theta0 + (theta1 * km_count[j])
            error = predicted_price - price[j]
            sum_error_theta0 += error
            sum_error_theta1 += error * km_count[j]
        
        d_theta0 = (1/m) * sum_error_theta0
        d_theta1 = (1/m) * sum_error_theta1

        theta0 -= alpha * d_theta0
        theta1 -= alpha * d_theta1

        # Optionnel : Suivre la convergence en affichant le coût
        cost = compute_cost([theta0 + theta1 * km_count[j] for j in range(m)], price)
        # if i % 100 == 0:  # Affiche le coût toutes les 100 itérations
        #     print(f"Iteration {i+1}, Cost: {cost:.6f}")

    return theta0, theta1

# BONUS FCT #######################
from sklearn.metrics import r2_score
def calculate_precision(km, price, theta0, theta1):
    m = len(km)
    predicted_prices = [theta0 + theta1 * x for x in km]
    # Calculer l'erreur quadratique moyenne (MSE)
    mse = sum((predicted_prices[i] - price[i]) ** 2 for i in range(m)) / m
    print(f"Erreur quadratique moyenne (MSE) : {mse:.4f}")

    return predicted_prices


def model(X, theta0, theta1):
    return [(theta0 + (theta1 * x)) for x in X]


def plot_data_and_regression_line(km, price, predict):
    plt.scatter(km, price, color='blue', label='Données')
    plt.plot(km, predict, color='red', label='Ligne de régression')
    plt.xlabel('Kilométrage')
    plt.ylabel('Prix')
    plt.title('Régression Linéaire')
    plt.legend()
    plt.show()
# #################################

# Enregistrer les parametres theta0 et theta1 
def save_params(theta0, theta1, file_path='ressources/params_model.txt'):
    with open(file_path, 'w') as file:
        file.write(f"{theta0},{theta1}")

def main():
    # Charger les données
    km, price = load_data()

    # Initialiser les paramètres
    theta0, theta1 = 0, 0
    alpha = 0.01
    num_iterations = 10000

    # Normaliser les données
    norm_km, km_min, km_max = normalize(km)
    norm_price, price_min, price_max = normalize(price)



    #print("norm_mileage:", norm_km, "mileage_min:", km_min, "mileage_max:", km_max)
    #print("norm_price:", norm_price, "price_min:", price_min, "price_max:", price_max)

    # Entraîner le modèle
    theta0, theta1 = do_gradient_descent(norm_km, norm_price, theta0, theta1, alpha, num_iterations)

    print("theta0:", theta0, "theta1:", theta1)

    # Denormaliser les paramètres theta0 et theta1 pour les sauvegarder
    # theta1 = theta1 * (price_max - price_min) / (km_max - km_min)

    # Sauvegarder les paramètres
    save_params(theta0, theta1)

    X_ok = denormalize(np.array(norm_km), km_min, km_max)
    y_ok = denormalize(np.array(norm_price), price_min, price_max)

    # Tracer les données et la ligne de régression

    # predictions = model(norm_km, )

    # Calculer la précision
    prediction = calculate_precision(norm_km, norm_price, theta0, theta1)
    print(f'PREDICTION BEFORE PLOT: {denormalize(prediction, price_min, price_max)}')
    plot_data_and_regression_line(X_ok, y_ok, denormalize(np.array(prediction), price_min, price_max))

    print(f'R2_Score: {r2_score(norm_price, prediction)}')
    print(f"Entraînement terminé. theta0 = {theta0:.12f}, theta1 = {theta1:.12f}")

if __name__ == "__main__":
    main()