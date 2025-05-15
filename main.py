# Importación de librerías necesarias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# --- Generación de Datos Sintéticos ---
# En un escenario real, obtendrías estos datos de fuentes financieras.
# Para este ejemplo, generaremos datos aleatorios.
np.random.seed(42) # Para reproducibilidad

num_bonos = 200

# Características de los bonos (variables independientes)
# Tasa de cupón (%)
tasa_cupon = np.random.uniform(1, 8, num_bonos)
# Tiempo hasta el vencimiento (años)
vencimiento = np.random.uniform(1, 30, num_bonos)
# Calificación crediticia (numérica, donde más alto es mejor, ej. 1=AAA, 2=AA, ..., 7=C)
# Simplificaremos con una escala numérica continua para el ejemplo
calificacion_num = np.random.uniform(1, 7, num_bonos)
# Tasa de interés de referencia del mercado (%) - Factor macroeconómico
tasa_mercado_ref = np.random.uniform(0.5, 5, num_bonos)
# Volatilidad del mercado (índice simplificado)
volatilidad = np.random.uniform(0.1, 0.5, num_bonos)

# Precio del bono (variable dependiente) - Lo que queremos predecir
# El precio de un bono generalmente se mueve inversamente a las tasas de interés.
# Bonos con cupones más altos o mejor calificación tienden a ser más caros.
# Esta es una fórmula simplificada para generar precios sintéticos.
precio_bono = 100 \
              + (tasa_cupon - tasa_mercado_ref) * 10 \
              - (vencimiento / 5) \
              - (calificacion_num * 2) \
              + np.random.normal(0, 5, num_bonos) # Ruido aleatorio

# Creación de un DataFrame de Pandas
data = pd.DataFrame({
    'TasaCupon': tasa_cupon,
    'Vencimiento': vencimiento,
    'CalificacionNum': calificacion_num,
    'TasaMercadoRef': tasa_mercado_ref,
    'Volatilidad': volatilidad,
    'PrecioBono': precio_bono
})

print("--- Primeras filas de los datos generados ---")
print(data.head())
print("\n--- Descripción estadística de los datos ---")
print(data.describe())

# --- Visualización Exploratoria de Datos (Opcional pero Recomendado) ---
# sns.pairplot(data, diag_kind='kde')
# plt.suptitle("Pairplot de las Características y el Precio del Bono", y=1.02)
# plt.show()

# plt.figure(figsize=(10, 6))
# sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
# plt.title("Mapa de Calor de Correlaciones")
# plt.show()

# --- Preparación de los Datos para el Modelo ---
# Seleccionamos las características (X) y la variable objetivo (y)
X = data[['TasaCupon', 'Vencimiento', 'CalificacionNum', 'TasaMercadoRef', 'Volatilidad']]
y = data['PrecioBono']

# Dividimos los datos en conjuntos de entrenamiento y prueba
# test_size=0.2 significa que el 20% de los datos se usarán para prueba
# random_state es para asegurar que la división sea la misma cada vez que se ejecuta el script
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nForma de X_train: {X_train.shape}")
print(f"Forma de X_test: {X_test.shape}")
print(f"Forma de y_train: {y_train.shape}")
print(f"Forma de y_test: {y_test.shape}")

# --- Entrenamiento del Modelo de Machine Learning ---
# Usaremos un modelo de Regresión Lineal simple como ejemplo.
# En la práctica, podrías probar modelos más complejos.
modelo = LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
modelo.fit(X_train, y_train)

print("\n--- Modelo Entrenado ---")
print(f"Coeficientes del modelo: {modelo.coef_}")
print(f"Intercepto del modelo: {modelo.intercept_}")

# --- Realización de Predicciones ---
# Predecir los precios de los bonos en el conjunto de prueba
y_pred = modelo.predict(X_test)

# --- Evaluación del Modelo ---
# Comparamos las predicciones (y_pred) con los valores reales (y_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2_score = modelo.score(X_test, y_test) # R-cuadrado

print("\n--- Evaluación del Modelo ---")
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.2f}")
print(f"Coeficiente de Determinación (R^2): {r2_score:.2f}")

# --- Visualización de Resultados (Opcional) ---
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='w', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2) # Línea de y=x
plt.xlabel("Precios Reales")
plt.ylabel("Precios Predichos")
plt.title("Precios Reales vs. Precios Predichos")
plt.grid(True)
plt.show()

# --- Ejemplo de Predicción para un Nuevo Bono (Hipotético) ---
nuevo_bono_caracteristicas = pd.DataFrame({
    'TasaCupon': [3.5],       # Tasa de cupón del 3.5%
    'Vencimiento': [10],      # 10 años hasta el vencimiento
    'CalificacionNum': [2],   # Calificación 'AA' (representada como 2)
    'TasaMercadoRef': [2.0],  # Tasa de mercado de referencia actual del 2.0%
    'Volatilidad': [0.25]     # Volatilidad del mercado de 0.25
})

prediccion_nuevo_bono = modelo.predict(nuevo_bono_caracteristicas)
print("\n--- Predicción para un Nuevo Bono Hipotético ---")
print(f"Características del nuevo bono:\n{nuevo_bono_caracteristicas}")
print(f"Precio predicho para el nuevo bono: {prediccion_nuevo_bono[0]:.2f}")

print("\n--- Fin del Script ---")
