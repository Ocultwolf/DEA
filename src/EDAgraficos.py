

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
import validators



class EDAgraficos():

    def __init__(self):

        url = input('''
        Hola, Bienvenido Al EDA !!!
                    
                    
            1. Carga De datos y previsualizado de la informcion
                    
                -Ingrese directorio o url :  ''')
        
        data = self.upload_data(url)

        
        print(f'''-Previsualizado: 
                        1.-las dimensiones del dataset son: {data.info()}
                        
                        2.- La informacion general es:
                           { data.info()}

                        3.-El dataset continue las siguientes columnas: {data.columns().tolist}

                        4.- 5 primemras filas:
                            {data.head()}''')  

        
        # data = self.clean_data(data)
        # graficos_dir= "-Ingrese directorio para la carpeta graficos(ej: ./graficos/)"
        
        # target=input(" -La target es (ej: price):")
        # graficos ='''
        #     3. Graficando graficos
        #         {input}
        #         -Graficando Frecuencias 
                
        #         -Graficando simetrias y outliers de las variables numericas
                
        #         {}'''
                

        # for column in data.columns:

        #     self.plot_frequency_column(data,column, graficos_dir)

        

        # self.Outliers = {}
        # self.dictionaries = {}  # Consolidated dictionaries
        

        # data_categoric = self.filter_column_type(data, 'categoric')
        # data_numeric = self.filter_column_type(data, 'numeric')
        
        # # Plot outliers and relationships for numeric columns
        # for column in data_numeric.columns:
        #     self.plot_outliers(column, data_numeric, graficos_dir)
        # # Plot relationships for categorical columns
        # for column in data_categoric.columns:
        #     if column != target:
        #         self.plot_categorical_relationships(data, target, column, graficos_dir)
        # # Transform categorical columns to numeric
        # for column in data_categoric.columns:
        #     self.data_transform = self.transform_categoric_in_numeric(data, column, diccionarios_dir)
        # # Generate correlation matrix and filter top pairs
        # result = self.correlation_matrix(self.data_transform, graficos_dir)
        # print(result)
        # df_filtrado = result[result["Correlación"] > 0.5]  # Filtering correlations above 0.5
        # # Create a dictionary for the filtered pairs
        # suma_variables = {
        #     f"Par_{i}": {"Variable 1": row["Variable 1"], "Variable 2": row["Variable 2"]}
        #     for i, row in df_filtrado.iterrows()
        # }
        # self.dictionaries["top_correlations"] = suma_variables
        # print(suma_variables)
        # # Save all dictionaries
        # self.save_dictionaries(diccionarios_dir)
        # # Generate pairplot for data
        # sns.pairplot(data=self.data_transform)
        # plt.savefig(os.path.join(self.graficos_dir, "pairplot.png"))
        # plt.close()
        # self.stats = self.data_transform.describe()
        # self.procesar_outliers()
    def upload_data(self, origen):
        
        def validar_entrada(argumento):
            while True: 
                if validators.url(argumento):
                    return "url"
                
                elif os.path.exists(argumento):
                    return "path"
                else:
                    argumento =  input(''' El valor introducido no coincide con un path o url valido.
                            intentelo de nuevo: ''')
    
        def es_extension_valida(ruta):
            extensiones_validas = {".csv"}
            _, extension = os.path.splitext(ruta)
            return extension.lower() in extensiones_validas
        
        tipo_origen = validar_entrada(origen)
        
        if tipo_origen == "url":
            # Descargar archivo desde la URL
            path_guardado = input(
        '''       
                -Ingrese el path donde quiere guardar el archivo descargado: ''')
            if not os.path.exists(os.path.dirname(path_guardado)):
                os.makedirs(os.path.dirname(path_guardado))
            try:
                response = requests.get(origen)
                response.raise_for_status()
                with open(path_guardado, "wb") as f:
                    f.write(response.content)
                print(
        f'''        
                    [+] Archivo descargado y guardado en: {path_guardado}''')
            except requests.RequestException as e:
                print(f"Error al descargar el archivo: {e}")
                raise
            archivo = path_guardado
        else:
            # Cargar archivo desde el path local
            archivo = origen

        # Verificar si la extensión es válida
        if not es_extension_valida(archivo):
            print("Error: Extensión no soportada. Solo se soportan archivos '.csv'.")
            return None

        # Intentar leer el archivo como DataFrame
        while True:
            try:
                df = pd.read_csv(archivo, sep=",")
                print(
        f'''        
                    [+] Archivo cargado exitosamente''')
                return df
            except pd.errors.ParserError as e:
                print(f"Error al cargar el archivo: {e}")
                nuevo_sep = input("El separador por defecto ',' parece incorrecto. Introduce el separador correcto: ")
                try:
                    df = pd.read_csv(archivo, sep=nuevo_sep)
                    print("Archivo cargado exitosamente con el nuevo separador.")
                    return df
                except Exception as e:
                    print(f"No se pudo cargar el archivo con el separador '{nuevo_sep}'. Error: {e}")
                    continuar = input("¿Quieres intentar con otro separador? (s/n): ").strip().lower()
                    if continuar != "s":
                        raise



    def clean_data(self, data):
        idcolumns = list(input("1.-Igrese Columnas identificadoras para obviar: "))
        columnas_a_eliminar = input("2.-Ingrese columnas a eliminar:")
        clean ='''

            2.- limpieza de datos
                -Eliminar columnas repetidas y 
                    {idcolumns}

                -Eliminar Columnas irrelevantes
                    {columnas_a_eliminar}

'''
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

    def  inputs(self,text):
        a = input(text)
        return a




# Bienvenida= '''

# Hola Bienvenida Al EDA

#     1. Ingrese la siguiente informacion al objeto()
# '''
# url = '''
#         path or url :'''
# nombre: '''dsd

# '''

# print(Bienvenida)
# url = input(url)
# EDAgraficos().upload_data(url)

EDAgraficos()

# mle = DLEMachineLearning(eda.data, target_column="target")
# mle.train_model()
# mle.evaluate_model()





