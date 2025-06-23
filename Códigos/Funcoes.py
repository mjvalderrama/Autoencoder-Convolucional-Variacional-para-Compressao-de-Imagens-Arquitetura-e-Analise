import numpy as np
import matplotlib.pyplot as plt

# Definição das funções
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# Criar um vetor de entrada para o exemplo
x = np.linspace(-4, 4, 400)

# Calcular as saídas
y_relu = relu(x)
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)

# Plotar os gráficos
plt.figure(figsize=(10, 6))

plt.plot(x, y_relu, label='ReLU', linewidth=3)
plt.plot(x, y_sigmoid, label='Sigmoid', linewidth=3)
plt.plot(x, y_tanh, label='Tanh', linewidth=3)

plt.title('Funções de Ativação: ReLU, Sigmoid e Tanh')
plt.xlabel('x', fontsize=20)
plt.ylabel('f(x)', fontsize=20)
plt.legend(fontsize=25)
plt.grid(True)
plt.tight_layout()
plt.show()
