# Seasonal SEIR Model for Dengue Transmission Dynamics
# Author: [Seu Nome]
# Federal Rural University of Pernambuco, Department of Statistics and Informatics

import numpy as np
import matplotlib.pyplot as plt

# ===== Parâmetros do modelo =====
N = 1000000   # população total
beta0 = 0.5   # taxa de transmissão base
gamma = 1/10  # taxa de recuperação (1/dias)
sigma = 1/5   # taxa de progressão de expostos para infectados (1/dias)

# Condições iniciais
I0 = 10       # infectados iniciais
E0 = 20       # expostos iniciais
R0 = 0        # recuperados iniciais
S0 = N - I0 - E0 - R0  # suscetíveis iniciais

# Tempo de simulação
dias = 3 * 365
t = np.linspace(0, dias, dias)

# ===== Função sazonal para β(t) =====
def beta_sazonal(t):
    """
    Função para taxa de transmissão sazonal β(t)
    Varia de forma senoidal ao longo do ano (365 dias)
    """
    return beta0 * (1 + 0.3 * np.sin(2 * np.pi * t / 365))

# Vetores para armazenar resultados
S, E, I, R = np.zeros(dias), np.zeros(dias), np.zeros(dias), np.zeros(dias)
S[0], E[0], I[0], R[0] = S0, E0, I0, R0

# ===== Simulação com Método de Euler =====
for day in range(1, dias):
    beta = beta_sazonal(day)
    
    dS = -beta * S[day-1] * I[day-1] / N
    dE = beta * S[day-1] * I[day-1] / N - sigma * E[day-1]
    dI = sigma * E[day-1] - gamma * I[day-1]
    dR = gamma * I[day-1]
    
    S[day] = S[day-1] + dS
    E[day] = E[day-1] + dE
    I[day] = I[day-1] + dI
    R[day] = R[day-1] + dR

# ===== Gráfico da evolução dos compartimentos =====
plt.figure(figsize=(10,6))
plt.plot(t, S, label="Susceptible")
plt.plot(t, E, label="Exposed")
plt.plot(t, I, label="Infectious")
plt.plot(t, R, label="Recovered")
plt.title("SEIR Model for Dengue with Seasonal Transmission Rate")
plt.xlabel("Days")
plt.ylabel("Number of individuals")
plt.legend()
plt.grid(True)
plt.show()

# ===== Gráfico da variação de β(t) =====
plt.figure(figsize=(8,4))
plt.plot(t, [beta_sazonal(day) for day in range(dias)], color='red')
plt.title("Seasonal Variation of β(t)")
plt.xlabel("Days")
plt.ylabel("β(t)")
plt.grid(True)
plt.show()