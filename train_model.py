import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def compute_cost(predicted_price, price):  
    m = len(price)
    return (1/(2*m)) * sum([(price[i] - predicted_price[i]) ** 2 for i in range(m)])

def load_data(file_path='ressources/data.csv'):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} est introuvable...")

    try:
        data = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Le fichier {file_path} est vide ou corrompu...")
    except pd.errors.ParserError:
        raise ValueError(f"Erreur de parsing du fichier {file_path}...")

    required_columns = ['km', 'price']
    if not all(col in data.columns for col in required_columns):
        raise KeyError(f"Les colonnes attendues {required_columns} ne correspondent pas...")

    data['km'] = pd.to_numeric(data['km'], errors='coerce')
    data['price'] = pd.to_numeric(data['price'], errors='coerce')

    data = data.dropna(subset=required_columns)

    km_count = data['km'].values
    price = data['price'].values

    print("Km Count - Min:", km_count.min(), "Max:", km_count.max(), "Moyenne:", km_count.mean())
    print("Price - Min:", price.min(), "Max:", price.max(), "Moyenne:", price.mean())

    return km_count, price

def normalize(data):
    min_val = min(data)
    max_val = max(data)
    if min_val == max_val:
        return [0] * len(data), min_val, max_val
    
    return [(x - min_val) / (max_val - min_val) for x in data], min_val, max_val

def denormalize(value, min_val, max_val):
    return [(x * (max_val - min_val)) + min_val for x in value]

def do_gradient_descent(km_count, price, theta0, theta1, alpha, iter_count):
    m = len(km_count)
    mse_history = [] 

    for i in range(iter_count):
        sum_error_theta0 = 0
        sum_error_theta1 = 0
        sum_squared_errors = 0

        for j in range(m):
            predicted_price = theta0 + (theta1 * km_count[j])
            error = predicted_price - price[j]
            sum_error_theta0 += error
            sum_error_theta1 += error * km_count[j]
            sum_squared_errors += error ** 2

        d_theta0 = (1/m) * sum_error_theta0
        d_theta1 = (1/m) * sum_error_theta1

        theta0 -= alpha * d_theta0
        theta1 -= alpha * d_theta1

        mse = (1/m) * sum_squared_errors
        mse_history.append(mse)

    return theta0, theta1, mse_history

def model(X, theta0, theta1):
    return [theta0 + (theta1 * x) for x in X]

def plot_data_and_mse(km, price, predict, mse_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.scatter(km, price, color='blue', label='Données')
    ax1.plot(km, predict, color='red', label='Ligne de régression')
    ax1.set_xlabel('Kilométrage')
    ax1.set_ylabel('Prix')
    ax1.set_title('Régression Linéaire')
    ax1.legend()

    ax2.plot(mse_history, color='green')
    ax2.set_xlabel('Itérations')
    ax2.set_ylabel('Mean Squared Error (MSE)')
    ax2.set_title('Évolution du MSE au Fil des Itérations')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def save_params(theta0, theta1, file_path='ressources/params_model.txt'):
    with open(file_path, 'w') as file:
        file.write(f"{theta0},{theta1}")

def main():
    km, price = load_data()

    theta0, theta1 = 0, 0
    alpha = 0.01
    num_iterations = 10000

    norm_km, km_min, km_max = normalize(km)
    norm_price, price_min, price_max = normalize(price)

    theta0, theta1, mse_history = do_gradient_descent(norm_km, norm_price, theta0, theta1, alpha, num_iterations)

    theta1_denom = theta1 * (price_max - price_min) / (km_max - km_min)
    theta0_denom = price_min + theta0 * (price_max - price_min) - theta1_denom * km_min

    save_params(theta0_denom, theta1_denom)

    prediction = model(norm_km, theta0, theta1)
    prediction_denom = denormalize(prediction, price_min, price_max)

    X_ok = denormalize(norm_km, km_min, km_max)
    y_ok = denormalize(norm_price, price_min, price_max)

    plot_data_and_mse(X_ok, y_ok, prediction_denom, mse_history)

    print(f'R2_Score: {r2_score(y_ok, prediction_denom)}')
    print(f"Entraînement terminé. theta0 = {theta0_denom:.12f}, theta1 = {theta1_denom:.12f}")

if __name__ == "__main__":
    main()
