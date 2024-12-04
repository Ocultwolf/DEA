from utils import db_connect
engine = db_connect()

# your code here


import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import numpy as np



class EDAgraficos:
    def _init_(self, url, nombre, directorio, idcolumns, columnas_a_eliminar, graficos_dir, target, diccionarios_dir):
        self.Outliers = {}
        self.dictionaries = {}  # Consolidated dictionaries
        self.graficos_dir = graficos_dir
        # Download and load the data
        data = self.upload_data(url, nombre, directorio)
        self.info(data)
        # Clean data and categorize columns
        data = self.clean_data(data, idcolumns, columnas_a_eliminar)
        data_categoric = self.filter_column_type(data, 'categoric')
        data_numeric = self.filter_column_type(data, 'numeric')
        # Plot frequency for all columns
        for column in data.columns:
            self.plot_frequency_column(data, column, graficos_dir)
        # Plot outliers and relationships for numeric columns
        for column in data_numeric.columns:
            self.plot_outliers(column, data_numeric, graficos_dir)
            if column != target:
                self.plot_column_relationship(data, target, column, graficos_dir)
        # Plot relationships for categorical columns
        for column in data_categoric.columns:
            if column != target:
                self.plot_categorical_relationships(data, target, column, graficos_dir)
        # Transform categorical columns to numeric
        for column in data_categoric.columns:
            self.data_transform = self.transform_categoric_in_numeric(data, column, diccionarios_dir)
        # Generate correlation matrix and filter top pairs
        result = self.correlation_matrix(self.data_transform, graficos_dir)
        print(result)
        df_filtrado = result[result["Correlación"] > 0.5]  # Filtering correlations above 0.5
        # Create a dictionary for the filtered pairs
        suma_variables = {
            f"Par_{i}": {"Variable 1": row["Variable 1"], "Variable 2": row["Variable 2"]}
            for i, row in df_filtrado.iterrows()
        }
        self.dictionaries["top_correlations"] = suma_variables
        print(suma_variables)
        # Save all dictionaries
        self.save_dictionaries(diccionarios_dir)
        # Generate pairplot for data
        sns.pairplot(data=self.data_transform)
        plt.savefig(os.path.join(self.graficos_dir, "pairplot.png"))
        plt.close()
        self.stats = self.data_transform.describe()
        self.procesar_outliers()
    def upload_data(self, url, nombre, directorio):
        if not os.path.exists(directorio):
            os.makedirs(directorio)
        ruta_completa = os.path.join(directorio, nombre)
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error al descargar el archivo: {e}")
            raise
        with open(ruta_completa, "wb") as f:
            f.write(response.content)
        return pd.read_csv(ruta_completa, sep=",")
    def info(self, data):
        print(data.shape)
        print(data.info())
        print(data.columns)
        print(data.head())
    def clean_data(self, data, idcolumns, columnas_a_eliminar):
        data.drop_duplicates(subset=idcolumns, inplace=True)
        if columnas_a_eliminar:
            data.drop(columns=columnas_a_eliminar, inplace=True, errors="ignore")
        return data
    def plot_frequency_column(self, data, column, output_dir):
        freq = data[column].value_counts().sort_values(ascending=False)
        if not freq.empty:
            plt.figure(figsize=(12, 6))
            sns.barplot(x=freq.index, y=freq.values)
            plt.title(f"Frequency Distribution of {column}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{column}_frequency.png"))
            plt.close()
    def plot_outliers(self, column, data, output_dir):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=data[column], orient="h")
        plt.title(f"Outliers in {column}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{column}_outliers.png"))
        plt.close()
    def plot_column_relationship(self, data, target, column, output_dir):
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x=data[column], y=data[target])
        plt.title(f"Relationship between {target} and {column}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{target}vs{column}.png"))
        plt.close()
    def plot_categorical_relationships(self, data, target, column, output_dir):
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=data[column], y=data[target])
        plt.title(f"Categorical Relationship of {column} with {target}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{column}vs{target}.png"))
        plt.close()
    def transform_categoric_in_numeric(self, data, column):
        mapping = {val: idx for idx, val in enumerate(data[column].unique())}
        data[column + "_n"] = data[column].map(mapping)
        self.dictionaries[column] = mapping
        data.drop(columns=[column], inplace=True)
        return data
    def save_dictionaries(self, diccionarios_dir):
        with open(os.path.join(diccionarios_dir, "diccionarios.json"), "w") as f:
            json.dump(self.dictionaries, f)
    def correlation_matrix(self, data, output_dir):
        corr_matrix = data.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
        plt.close()
        # Extract top 10 correlations
        corr_pairs = (
            corr_matrix.where(
                lambda x: ~np.tril(np.ones(x.shape), k=0).astype(bool)
            )
            .stack()
            .abs()
            .sort_values(ascending=False)
        )
        top_10_pairs = corr_pairs.head(10).reset_index()
        top_10_pairs.columns = ['Variable 1', 'Variable 2', 'Correlación']
        return top_10_pairs
    def procesar_outliers(self, k=1.5):
        for columna in self.data_transform.columns:
            Q1 = self.data_transform[columna].quantile(0.25)
            Q3 = self.data_transform[columna].quantile(0.75)
            IQR = Q3 - Q1
            upper_limit = Q3 + k * IQR
            lower_limit = Q1 - k * IQR
            self.Outliers[columna] = {'IQR': IQR, 'upper_limit': upper_limit, 'lower_limit': lower_limit}









eda = EDAgraficos(data_path="marketingbancario.csv", graficos_dir="./graficos", target_column="target", diccionarios_dir="./DiccionarioCategorias")
eda.data.to_csv("processed_data.csv", index=False)  # Save processed data for ML

mle = DLEMachineLearning(eda.data, target_column="target")
mle.train_model()
mle.evaluate_model()





