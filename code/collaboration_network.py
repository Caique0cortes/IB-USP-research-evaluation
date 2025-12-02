# -*- coding: utf-8 -*-


import pandas as pd
import itertools
from itertools import combinations
import plotly.graph_objects as go
import networkx as nx
import re



!pip install networkx

import networkx as nx

#File path
base_geral = pd.read_excel('General database.xlsx')

#Identifying related researchers
Lista_pesquisadores_em_comum = []
for x, frame in base_geral.groupby(['Titulo do artigo']):
  Lista_pesquisadores_em_comum.append(list(frame['Pesquisador']))

#Edges are defined by the combination of the set of authors who share a given publication in common
edges = [] #Generates a list of lists in the course of the task

for colaboracao in Lista_pesquisadores_em_comum:
  edges.append(list(combinations(colaboracao,2)))

edges = list(itertools.chain(*edges)) #Transforms the list of lists into a single list

#Creating a dictionary that indicates the number of times each relationship occurs
relacoes_unicas = list(set(edges))
dic_relacionamentos = {}

#Creating a dictionary for each relationship found
for relacionamento in relacoes_unicas:
  dic_relacionamentos.update({relacionamento:0})
#Updating the dictionary with the number of times the relationship occurs
for i in edges:
  dic_relacionamentos[i] +=1



#Cleaning the dictionary by removing individuals who are related to themselves, based on the creation of a new dictionary of edges and weights
novo_dicionario = {}
auto_relacionamento = {}
for i in dic_relacionamentos.keys():
  if i[0] != i[1]:
    novo_dicionario.update({i: dic_relacionamentos[i]})
  else:
    auto_relacionamento.update({i: dic_relacionamentos[i]})


#Defining who the nodes of the network are
indivíduos_com_relacionamento = []

for i in novo_dicionario.keys():
  for nome in i:
    indivíduos_com_relacionamento.append(nome)
nodos = list(set(indivíduos_com_relacionamento))

#Defining the color of the departments

cor_departamento ={'Zoologia': 'bisque',
                   'Botânica' : 'lightgreen',
                   'Fisiologia': 'salmon',
                   'Genética': 'turquoise',
                   'Ecologia': 'teal'}


#Size and color of the nodes
dic_nodos = {}
for nodo in nodos:
  tamanho = len(re.findall(nodo, str(list(novo_dicionario.keys()))))
  departamento = list(base_geral.loc[base_geral['Pesquisador'] == nodo]['Departamento'].unique())[0]
  color = cor_departamento[str(departamento)]


  dic_nodos.update({nodo: [tamanho, color]})



#Organizing the dictionary data into lists [individual1, individual2, relationship weight]
relacao_peso = []
for i in novo_dicionario:
  relacao_peso.append([i[0], i[1], dic_relacionamentos[i]])

#Creating a graph
G = nx.Graph()
for node in dic_nodos.keys():
  G.add_node(node, size = dic_nodos[node][0], departamento = dic_nodos[node][1])

for edge in relacao_peso:
  G.add_edge(edge[0],edge[1], weight= edge[2])
pos_ = nx.fruchterman_reingold_layout(G)


#Creating the edge_trace

edge_trace =[]
def get_trace_edge(x,y,width): #A function so that each edge has a coordinate, a weight, and a color

  if width < 10:
    color = 'LightSkyBlue'
  else:
    color = 'SkyBlue'

  return go.Scatter(x = x,
                    y= y,
                    line = dict(width = width*0.3, color =color),
                    hoverinfo = 'text',
                    mode = 'lines')

for edge in G.edges(): #Takes a relationship between a pair of individuals

  Individuo0 = edge[0] #The x and y coordinates in the graph for the edge, given the connection with the first individual.
  Individuo1 = edge[1]
  x0,y0 = pos_[Individuo0] #Defining the variables for each x and y coordinate
  x1,y1 = pos_[Individuo1]

  trace_edge = get_trace_edge([x0,x1],[y0,y1], width = list((G.edges()[edge]).values())[0]) #traçando a personalização da edge
  edge_trace.append(trace_edge) #Adding the defined pattern for each edge to a list

# Creating the node_trace

def get_trace_nodo(x,y,size, collor):

  return go.Scatter(x = x,
                    y = y,
                    mode = 'markers',
                    marker=dict(size = size,
                                color = collor,
                                line=None)
  )


trace_nodo_size =[]
trace_nodo_collor = []
trace_nodo_x=[]
trace_nodo_y=[]
for node in G.nodes():
  x,y = tuple(pos_[node])
  trace_nodo_x.append(x)
  trace_nodo_y.append(y)
  trace_nodo_size.append(list((G.nodes()[node]).values())[0])
  trace_nodo_collor.append(list((G.nodes()[node]).values())[1])


nodo_trace = get_trace_nodo(trace_nodo_x,trace_nodo_y,trace_nodo_size, trace_nodo_collor)

layout = go.Layout(
    paper_bgcolor='black',
    plot_bgcolor='black',
    xaxis =  {'showgrid': False, 'zeroline': False}, # no gridlines
    yaxis = {'showgrid': False, 'zeroline': False}, # no gridlines
)
# Create figure
fig = go.Figure(layout = layout)
# Add all edge traces
for trace in edge_trace:
    fig.add_trace(trace)
# Add node trace
fig.add_trace(nodo_trace)
# Remove legend
fig.update_layout(showlegend = False)
# Remove tick labels
fig.update_xaxes(showticklabels = False)
fig.update_yaxes(showticklabels = False)


#Configuring the legend settings

legend_trace = go.Scatter(
    x=[0.1, 0.1, 0.1, 0.1, 0.1],
    y=[0.62, 0.74, 0.86, 0.98, 1.1],
    mode='markers',
    marker=dict(
        size=10,
        color=list(set(trace_nodo_collor)),  # Unique colors of the departments
        showscale=False
    ),
    showlegend=True,
    legendgroup='Legenda',
    hoverinfo='text',
)


#Function to define the department by the legend color.
cores = legend_trace['marker']['color']
def achar_departamento(cor):
    for chave, valor in cor_departamento.items():
        if valor == cor:
            return chave

# Adding the legend and the legend text.

fig.add_trace(legend_trace) #Inserts the colored dots

Legenda_text = {
    'x': [x+0.02 for coordenada in legend_trace['x']],
    'y': legend_trace['y'],
    'Departamento': [achar_departamento(cor) for cor in cores]
}


for x, y, departamento in zip(Legenda_text['x'], Legenda_text['y'], Legenda_text['Departamento']):
    fig.add_annotation(
        x=x,
        y=y,
        text=departamento,
        showarrow=False,
        font=dict(color='White', size=15))


fig.show()
