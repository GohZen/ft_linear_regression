# predict_price.py

# Charger les paramètres theta0 et theta1 depuis le fichier
def load_params(file_path='ressources/params_model.txt'):
    with open(file_path, 'r') as file:
        theta0, theta1 = map(float, file.readline().strip().split(','))
    return theta0, theta1

# Estimer le prix basé sur le kilométrage et les paramètres theta0 et theta1
def estimate_price(number_of_miles, theta0, theta1):
    return theta0 + (theta1 * number_of_miles)

def main():
    # Charger les paramètres theta0 et theta1
    theta0, theta1 = load_params()

    # Demander à l'utilisateur d'entrer le kilométrage
    mileage = float(input("Entrez le kilométrage de la voiture: "))

    # Calculer le prix estimé
    estimated_price = estimate_price(mileage, theta0, theta1)

    # Afficher le prix estimé
    print(f"Le prix estimé pour un kilométrage de {mileage} km est de {estimated_price:.2f} euros.")

if __name__ == "__main__":
    main()
