import tkinter as tk
from tkinter import messagebox
import numpy as np

# Función de activación (sigmoide)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función sigmoide
def sigmoid_derivative(x):
    return x * (1 - x)

# Propagación hacia adelante
def forward_propagation(X, W, b):
    z = np.dot(X, W) + b
    y = sigmoid(z)
    return y

# Retropropagación
def backpropagation(X, y, d, W, b, alpha):
    error = d - y
    dW = alpha * np.dot(X.T, error * sigmoid_derivative(y))
    db = alpha * np.sum(error * sigmoid_derivative(y))
    return dW, db

# Entrenamiento
def train(X, d, epochs, alpha):
    n, m = X.shape  # n: número de muestras, m: número de características
    W = np.random.uniform(-1, 1, (m, 1))  # Inicialización aleatoria de pesos
    b = np.random.uniform(-1, 1, (1,))    # Inicialización aleatoria del sesgo

    for _ in range(epochs):
        y = forward_propagation(X, W, b)  # Propagación hacia adelante
        dW, db = backpropagation(X, y, d, W, b, alpha)  # Retropropagación
        W += dW  # Actualización de pesos
        b += db  # Actualización del sesgo

    return W, b

# Función para mostrar mensajes de error
def show_error(message):
    messagebox.showerror("Error", message)

# Función para mostrar mensajes de información
def show_info(message):
    messagebox.showinfo("Información", message)

# Función para entrenar el modelo
def train_model():
    try:
        # Validación de entradas para X (separado por punto y coma y comas)
        X_input = entry_x.get()
        if not X_input:
            show_error("Por favor, ingrese las entradas para X.")
            return
        
        X = np.array([[float(x) for x in row.split(",")] for row in X_input.split(";")])

        # Validación de las salidas deseadas (d)
        d_input = entry_d.get()
        if not d_input:
            show_error("Por favor, ingrese las salidas deseadas (d).")
            return
        
        d = np.array([float(x) for x in d_input.split(",")])

        # Asegurarse de que las dimensiones coinciden
        if X.shape[0] != len(d):
            show_error("El número de entradas no coincide con el número de salidas.")
            return

        # Convertir las salidas deseadas d en una matriz de columna
        d = d.reshape(-1, 1)

        # Validación de las épocas y tasa de aprendizaje
        try:
            epochs = int(entry_epochs.get())
            alpha = float(entry_alpha.get())
        except ValueError:
            show_error("Por favor, ingrese números válidos para las épocas y la tasa de aprendizaje.")
            return

        if epochs <= 0 or alpha <= 0:
            show_error("Las épocas y la tasa de aprendizaje deben ser mayores a cero.")
            return

        # Entrenar el modelo
        W, b = train(X, d, epochs, alpha)

        # Mostrar los resultados
        result_label.config(text=f"Pesos finales (W): {W.flatten()}\nBias final (b): {b}")

    except ValueError:
        show_error("Por favor, ingrese datos válidos. Asegúrese de que las entradas sean números.")

# Crear ventana principal
root = tk.Tk()
root.title("Algoritmo de Retropropagación")
root.geometry("500x400")  # Ajusta el tamaño de la ventana
root.config(bg="#f7f7f7")  # Color de fondo

# Frame principal
frame = tk.Frame(root, bg="#f7f7f7")
frame.pack(padx=20, pady=20)

# Etiquetas y entradas
label_x = tk.Label(frame, text="Entradas (X):\nFormato: 1,2,3; 4,5,6", bg="#f7f7f7", font=("Arial", 10, "bold"))
label_x.grid(row=0, column=0, sticky="w", pady=5)
entry_x = tk.Entry(frame, font=("Arial", 10), width=30)
entry_x.grid(row=0, column=1, pady=5)

label_d = tk.Label(frame, text="Salidas deseadas (d):\nFormato: 0,1,0", bg="#f7f7f7", font=("Arial", 10, "bold"))
label_d.grid(row=1, column=0, sticky="w", pady=5)
entry_d = tk.Entry(frame, font=("Arial", 10), width=30)
entry_d.grid(row=1, column=1, pady=5)

label_epochs = tk.Label(frame, text="Épocas:", bg="#f7f7f7", font=("Arial", 10, "bold"))
label_epochs.grid(row=2, column=0, sticky="w", pady=5)
entry_epochs = tk.Entry(frame, font=("Arial", 10), width=30)
entry_epochs.grid(row=2, column=1, pady=5)

label_alpha = tk.Label(frame, text="Tasa de aprendizaje (α):", bg="#f7f7f7", font=("Arial", 10, "bold"))
label_alpha.grid(row=3, column=0, sticky="w", pady=5)
entry_alpha = tk.Entry(frame, font=("Arial", 10), width=30)
entry_alpha.grid(row=3, column=1, pady=5)

# Botón de entrenamiento
button_train = tk.Button(frame, text="Entrenar", command=train_model, font=("Arial", 10, "bold"), bg="#4CAF50", fg="white", padx=10, pady=5)
button_train.grid(row=4, column=0, columnspan=2, pady=20)

# Etiqueta de resultados
result_label = tk.Label(frame, text="", bg="#f7f7f7", font=("Arial", 10))
result_label.grid(row=5, column=0, columnspan=2)

# Ejecutar la interfaz
root.mainloop()
