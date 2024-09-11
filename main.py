# OBJECTIFS DU PROGRAMME
#
# PREDIRE LE PIRX D UNE VOITURE EN UTILISANT UNE FONCTION DE
# REGRESSION LINEAIRE ENTRAINEE PAR UN ALGORITHME DE DECENTE
# DE GRADIENT
#
# -> Regression linéaire a une seule feature : le kilometrage
# -> Besoin de 2 programmes distinct : 
#       1. Predit le prix de la voiture en fonction du kilométrage.
#          Une fois lancé, le programme demande un kilométrage, et renvoie
#          le prix estimé pour ce kilométrage.
#          hypothèse de prédiction : estimatePrice(mileage) = θ0 + (θ1 ∗ mileage)
#          Avant de lancer le programme d'entrainement, theta0 et theta1 seront set a 0.
#
#       2. Sert a entrainer le modèle.
#          Lit le dataset dans un premier temps, puis effectue la regression linéaire sur les data.
#          Une fois cela effectué, il faut sauvegarder theta0 et theta1 pour utilisation avec le programme 1.
#          Formule a utiliser : <formule_progr_2.png>
#          A noter que estimatePrice est parei que dans le premier programme mais utilise les variables temporaires
#          theta0 et theta1. Il ne faut pas oublier de les mettre a jour simultanément.
# 
# BONUS PART
# 
# -> Tracer les données dans un graphique pour voir leur répartition.
# -> Tracez la droite résultant de votre régression linéaire dans le même graphique, pour voir le résultat de votre travail acharné !
# -> Un programme qui calcule la précision de votre algorithme.
# 

import os

def train_model():
    os.system('python train_model.py')

def predict_price():
    os.system('python predict_price.py')

def main():
    print("Sélectionnez une option :")
    print("1. Entraîner le modèle")
    print("2. Prédire le prix d'une voiture")

    while(True):
        choice = input("Votre choix (1, 2 ou 'exit'): ")
        if choice == '1':
            train_model()
        elif choice == '2':
            predict_price()
        elif choice == "exit":
            break
        else:
            print("Choix invalide. Veuillez sélectionner 1 ou 2.")

if __name__ == "__main__":
    main()