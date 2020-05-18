import numpy
import matplotlib

import pandas

import geopandas
import geoplot

#import datetime

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split


# Exibir gráficos:
#TODO exibir gráficos de boas
#%matplotlib
def show():
   return matplotlib.pyplot.show(block=True)

#Carregar mapa mundi
world = geopandas.read_file(
    geopandas.datasets.get_path('naturalearth_lowres')
)
geoplot.polyplot(world)
#TODO implementar download automático do Dataset

#Carrega dataset:
deaths = pandas.read_csv("total_deaths.csv")

# Filtrar as tabelas para evitar países não oficiais ou sem dados
lista_para_remover = ['Lesotho', 'eSwatini', 'Solomon Is.','Timor-Leste', 'Antarctica', 'Fr. S. Antarctic Lands','Vanuatu', "Côte d\'Ivoire", 'Turkmenistan', 'N. Cyprus', 'North Korea', 'Somaliland']
for i in lista_para_remover:
    world = world[~world["name"].str.contains(i)]

#Determina o número de mortos atualizado:
numero_mortos = pandas.Series.max(deaths[world["name"]])

#Ainda não achei maneira melhor de adicionar serie no dataframe:
world['numero_mortos'] = pandas.Series.tolist(numero_mortos) 

# Plota o gráfico:
world.plot(column='numero_mortos', legend=True, k=10)
 
 #Escolhe um país:
 matplotlib.pyplot.plot(x,alvos) 
  
#Determina o vetor de alvos:
alvos = pandas.DataFrame.to_numpy(deaths[["Brazil"]],dtype=numpy.double)[:,0]
alvos = alvos[~numpy.isnan(alvos)]
x = numpy.arange(alvos.size).reshape(-1, 1)


#Plota os dados do país selecionado:
matplotlib.pyplot.plot(x,alvos)   

#Divide os dados entre teste e treinamento:
x_train, x_test, alvos_train, alvos_test = train_test_split(x, alvos, test_size=0.3)

#Cria a rede neural:
rede = MLPRegressor(hidden_layer_sizes=(10, ), activation='tanh', solver='lbfgs', max_iter)

#Treina a rede:
rede.fit(x_train, alvos_train)

#Compara os resultados:
alvos_test_predict = rede.predict(x_test)

erros = alvos_predict - alvos_test
matplotlib.pyplot.hist(erros)

rede.score(x_test,alvos_test)
rede.n_iter_
rede.loss_

forecast = rede.predict(x)


Sabado
#Jupyter Notebook

Domingo
#Texto
