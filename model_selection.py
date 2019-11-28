# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 08:57:40 2019

@author: aalmoaia
"""

# Carregando Bibliotecas Python
import numpy as np
import pandas as pd
import os

from sklearn.linear_model import LinearRegression, Ridge, LassoCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
import warnings
warnings.filterwarnings("ignore")

# Função para calcular o RMSE
def rmse_cv(modelo):
    rmse = np.sqrt(-cross_val_score(modelo, 
                                    X_train, 
                                    y_train, 
                                    scoring = "neg_mean_squared_error", 
                                    cv = 5))
    return(rmse)
    

dataset = pd.read_csv('base.csv', index_col=0)

dataset = dataset.dropna()

val_selection = ['sum_nivel_pn3', 'IT01@TR133K01','WT02@TR133K08', 'WT02@TR133K01', 'WT02@TR133K04', 'sum_nivel_br3']

# Coletando x e y
# Usaremos como variáveis explanatórias somente as 4 variáveis mais relevantes
X = dataset[val_selection]
y = dataset['sum_alimentador_pn3'].values

# Divisão em dados de treino e de teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
standardization = StandardScaler()

# Cria o modelo Linear Multiplo
modelo_lr = LinearRegression(normalize = False, fit_intercept = True)
rmse_modelo_lr = rmse_cv(modelo_lr).mean()
r2_score_lr = r2_score(y_test, make_pipeline(standardization,modelo_lr).fit(X_train, y_train).predict(X_test))
print('Modelo Linear Multiplo')
print('RMSE treino = ',rmse_modelo_lr)
print('R2 teste = ', r2_score_lr)

# Cria o modelo Ridge
modelo_ridge = Ridge(alpha = 14500)
rmse_modelo_ridge = rmse_cv(modelo_ridge).mean()
r2_score_ridge = r2_score(y_test, make_pipeline(standardization,modelo_ridge).fit(X_train, y_train).predict(X_test))
print('\nModelo Ridge')
print('RMSE treino = ',rmse_modelo_ridge)
print('R2 teste = ', r2_score_ridge)

# Cria o modelo LASSO
modelo_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005])
rmse_modelo_lasso = rmse_cv(modelo_lasso).mean()
print('\nModelo Lasso')
print('RMSE treino = ',rmse_modelo_lasso)
r2_score_lasso = r2_score(y_test, make_pipeline(standardization,modelo_lasso).fit(X_train, y_train).predict(X_test))
print('R2 teste = ', r2_score_lasso)

#Modelo KNN
modelo_knn = KNeighborsRegressor(5,'uniform')
rmse_modelo_knn = rmse_cv(modelo_knn).mean()
r2_score_knn = r2_score(y_test, make_pipeline(standardization,modelo_knn).fit(X_train, y_train).predict(X_test))
print('\nModelo KNN')
print('RMSE treino = ',rmse_modelo_knn)
print('R2 teste = ', r2_score_knn)

#Modelo Elastic net
modelo_en = ElasticNet(random_state = 42)
rmse_modelo_en = rmse_cv(modelo_en).mean()
r2_score_en = r2_score(y_test, make_pipeline(standardization,modelo_en).fit(X_train, y_train).predict(X_test))
print('\nModelo ElasticNet')
print('RMSE treino = ',rmse_modelo_en)
print('R2 teste = ', r2_score_en)

#Polinominal
modelo_poly = Pipeline([('poly', PolynomialFeatures(degree=3)),
                        ('linear', LinearRegression(fit_intercept=False))])
rmse_modelo_poly = rmse_cv(modelo_poly).mean()
r2_score_poly = r2_score(y_test, make_pipeline(standardization,modelo_poly).fit(X_train, y_train).predict(X_test))
print('\nModelo Poly')
print('RMSE = ',rmse_modelo_poly)
print('R2 teste = ', r2_score_poly)

#Gradient Boosting Regressor
modelo_gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=0, loss='ls')
rmse_modelo_gbr = rmse_cv(modelo_gbr).mean()
r2_score_gbr = r2_score(y_test, make_pipeline(standardization,modelo_gbr).fit(X_train, y_train).predict(X_test))
print('\nModelo GBR')
print('RMSE = ',rmse_modelo_gbr)
print('R2 teste = ', r2_score_gbr)

#Voting Regressor
modelo_voting = VotingRegressor(estimators=[('gbr', modelo_gbr), ('poly', modelo_poly), ('knn', modelo_knn)])
rmse_modelo_voting = rmse_cv(modelo_voting).mean()
r2_score_voting = r2_score(y_test, make_pipeline(standardization,modelo_voting).fit(X_train, y_train).predict(X_test))
print('\nModelo Voting')
print('RMSE = ',rmse_modelo_voting)
print('R2 teste = ', r2_score_voting)