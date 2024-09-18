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