from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

app = Flask(__name__)

# Configuraciones
DATASET_PATH = 'ferreteria_dataset_large.csv'
STATIC_PATH = os.path.join('static', 'images')
os.makedirs(STATIC_PATH, exist_ok=True)

def load_and_preprocess_data():
    df = pd.read_csv(DATASET_PATH)
    
    # Convertir género a numérico
    df['Genero'] = df['Genero'].map({'F': 0, 'M': 1})
    
    # Detectar y manejar outliers
    numeric_cols = ['Edad', 'Precio', 'Cantidad', 'Dias_Desde_Ultima_Compra', 
                   'Total_Compras', 'Descuento_Aplicado']
    
    for col in numeric_cols:
        z_scores = np.abs(stats.zscore(df[col]))
        df[col] = np.where(z_scores > 3, df[col].median(), df[col])
    
    return df

@app.route('/pca')
def pca_analysis():
    # Cargar y preprocesar datos
    df = load_and_preprocess_data()
    
    # Características para PCA
    features = ['Edad', 'Genero', 'Precio', 'Cantidad', 
               'Dias_Desde_Ultima_Compra', 'Total_Compras', 'Descuento_Aplicado']
    X = df[features]
    
    # Usar RobustScaler para ser más resistente a outliers
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Aplicar PCA
    pca = PCA()
    principal_components = pca.fit_transform(X_scaled)
    
    # Crear visualizaciones mejoradas
    
    # 1. Scatter plot mejorado de los dos primeros componentes
    plt.figure(figsize=(12, 8))
    plt.clf()
    scatter_data = pd.DataFrame(data=principal_components[:, :2], 
                              columns=['PC1', 'PC2'])
    scatter_data['Compra_Futura'] = df['Compra_Futura']
    
    # Añadir clustering para mejor visualización
    kmeans = KMeans(n_clusters=3, random_state=42)
    scatter_data['Cluster'] = kmeans.fit_predict(scatter_data[['PC1', 'PC2']])
    
    # Crear scatter plot con clusters
    sns.scatterplot(data=scatter_data, x='PC1', y='PC2', 
                   hue='Compra_Futura', style='Cluster',
                   palette='viridis', alpha=0.6)
    
    plt.title('Análisis de Componentes Principales con Clustering')
    plt.legend(title='Probabilidad de Compra Futura', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_PATH, 'pca_scatter.png'))
    
    # 2. Gráfico de varianza explicada mejorado
    plt.figure(figsize=(12, 6))
    plt.clf()
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    # Crear subplot con dos gráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico de varianza individual
    ax1.bar(range(1, len(explained_variance_ratio) + 1), 
            explained_variance_ratio,
            alpha=0.5, color='b')
    ax1.plot(range(1, len(explained_variance_ratio) + 1), 
             explained_variance_ratio, 'ro-')
    ax1.set_xlabel('Componente Principal')
    ax1.set_ylabel('Proporción de Varianza Explicada')
    ax1.set_title('Varianza Explicada por Componente')
    ax1.grid(True)
    
    # Gráfico de varianza acumulada
    ax2.plot(range(1, len(cumulative_variance_ratio) + 1), 
             cumulative_variance_ratio, 'bo-')
    ax2.axhline(y=0.95, color='r', linestyle='--', 
                label='Umbral 95%')
    ax2.set_xlabel('Número de Componentes')
    ax2.set_ylabel('Varianza Explicada Acumulada')
    ax2.set_title('Varianza Explicada Acumulada vs Componentes')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_PATH, 'variance_explained.png'))
    
    # 3. Nuevo: Heatmap de correlaciones entre componentes y características
    plt.figure(figsize=(12, 8))
    plt.clf()
    loadings = pd.DataFrame(
        pca.components_.T * np.sqrt(pca.explained_variance_),
        columns=[f'PC{i+1}' for i in range(len(features))],
        index=features
    )
    sns.heatmap(loadings, annot=True, cmap='RdBu', center=0)
    plt.title('Correlaciones entre Características y Componentes Principales')
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_PATH, 'pca_correlations.png'))
    
    # Preparar datos adicionales para la template
    component_weights = pd.DataFrame(
        pca.components_[:2, :],
        columns=features,
        index=['PC1', 'PC2']
    ).round(3).to_dict('index')
    
    explained_variance = {
        'individual': explained_variance_ratio.round(3).tolist(),
        'cumulative': cumulative_variance_ratio.round(3).tolist()
    }
    
    # Calcular métricas adicionales
    feature_importance = pd.DataFrame({
        'Feature': features,
        'PC1_contribution': abs(pca.components_[0]) / sum(abs(pca.components_[0])),
        'PC2_contribution': abs(pca.components_[1]) / sum(abs(pca.components_[1]))
    }).round(3)
    
    # Calcular estadísticas de clusters
    cluster_stats = pd.DataFrame({
        'Cluster': range(3),
        'Size': scatter_data['Cluster'].value_counts().sort_index(),
        'Avg_PC1': scatter_data.groupby('Cluster')['PC1'].mean(),
        'Avg_PC2': scatter_data.groupby('Cluster')['PC2'].mean(),
        'Compra_Futura_Prob': scatter_data.groupby('Cluster')['Compra_Futura'].mean()
    }).round(3)
    
    return render_template(
        'pca.html',
        scatter_plot='images/pca_scatter.png',
        variance_plot='images/variance_explained.png',
        correlation_plot='images/pca_correlations.png',
        component_weights=component_weights,
        explained_variance=explained_variance,
        feature_importance=feature_importance.to_dict('records'),
        cluster_stats=cluster_stats.to_dict('records')
    )


class MultiArmedBandit:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_reward = 0.0
        self.history = []
    
    def select_arm(self, epsilon=0.1):
        if np.random.random() < epsilon:
            return int(np.random.randint(self.n_arms))
        return int(np.argmax(self.values))
    
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value
        self.total_reward += reward
        self.history.append({
            'arm': int(chosen_arm),
            'reward': float(reward),
            'cumulative_reward': float(self.total_reward)
        })



@app.route('/')
def index():
    return render_template('index.html')



@app.route('/bandit')
def bandit_analysis():
    # Cargar datos
    df = load_and_preprocess_data()
    
    # Simular diferentes estrategias de descuento como brazos del bandit
    descuentos = [0, 0.05, 0.10, 0.15, 0.20]
    n_simulaciones = 1000
    
    # Inicializar bandit
    bandit = MultiArmedBandit(len(descuentos))
    
    # Simular decisiones y recompensas
    for _ in range(n_simulaciones):
        arm = bandit.select_arm()
        descuento = descuentos[arm]
        
        similar_purchases = df[
            (df['Descuento_Aplicado'] >= descuento - 0.02) & 
            (df['Descuento_Aplicado'] <= descuento + 0.02)
        ]
        
        if len(similar_purchases) > 0:
            reward = float(similar_purchases['Compra_Futura'].mean())  # Convertir a float
        else:
            reward = 0.0
            
        bandit.update(arm, reward)
    
    # Convertir arrays numpy a listas Python y valores numpy a tipos Python nativos
    results = {
        'descuentos': descuentos,
        'valores_estimados': [float(v) for v in bandit.values],  # Convertir a float
        'veces_seleccionado': [int(c) for c in bandit.counts],   # Convertir a int
        'mejor_descuento': float(descuentos[np.argmax(bandit.values)]),  # Convertir a float
        'history': [
            {
                'arm': int(h['arm']),  # Convertir a int
                'reward': float(h['reward']),  # Convertir a float
                'cumulative_reward': float(h['cumulative_reward'])  # Convertir a float
            }
            for h in bandit.history[-100:]  # Últimas 100 iteraciones
        ]
    }
    
    return render_template('bandit.html', results=results)


if __name__ == '__main__':
    app.run(debug=True)
