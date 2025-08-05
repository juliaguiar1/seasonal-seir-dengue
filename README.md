# Seasonal SEIR Model for Dengue Transmission Dynamics
**Author:** Júlia A. C. Marinho  
**Institution:** Federal Rural University of Pernambuco, Department of Statistics and Informatics, Brazil  
**Email:** julia.aguiar@ufrpe.br  

---

## 📄 Project Description
This repository contains the implementation of a **SEIR (Susceptible–Exposed–Infectious–Recovered)** epidemiological model for **dengue transmission**, incorporating **seasonal variation** in the transmission rate β(t) to account for climatic effects such as temperature and rainfall.

The model is implemented in **Python** and simulates dengue outbreaks in a hypothetical population over a three-year period. The seasonal effect is modeled using a sinusoidal function, reflecting the annual climatic cycle and its influence on the *Aedes aegypti* mosquito population.

---

## 🛠 Features
- SEIR compartmental model
- Seasonal variation in β(t) using a sinusoidal function
- Numerical simulation using Euler's method
- Visualization of:
  - Compartment evolution (S, E, I, R)
  - Seasonal variation of β(t)
