import os

def load_params(file_path='ressources/params_model.txt'):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} est introuvable.")
    
    with open(file_path, 'r') as file:
        try:
            theta0, theta1 = map(float, file.readline().strip().split(','))
        except ValueError:
            raise ValueError("Erreur dans le format du fichier des paramètres.")
    
    return theta0, theta1

def estimate_price(number_of_miles, theta0, theta1):
    estimated_price = theta0 + (theta1 * number_of_miles)
    
    if estimated_price < 0:
        print(f"Attention : Le modèle prédit un prix négatif ({estimated_price:.2f} euros). "
            "Cela indique que le modèle est limité pour des kilométrages aussi élevés.")
        return 0
    
    return estimated_price

def main():
    try:
        theta0, theta1 = load_params()
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return

    try:
        mileage = float(input("Entrez le kilométrage de la voiture: "))
    except ValueError:
        print("Veuillez entrer un nombre valide pour le kilométrage.")
        return

    estimated_price = estimate_price(mileage, theta0, theta1)
    print(f"Le prix estimé pour un kilométrage de {mileage} km est de {estimated_price:.2f} euros.")

if __name__ == "__main__":
    main()
