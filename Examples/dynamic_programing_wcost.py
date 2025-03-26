import numpy as np
import time


class State:
    def __init__(self, cost=None, parents=None, children=None):
        self.cost = cost if cost is not None else []
        self.parents = parents if parents is not None else []
        self.children = children if children is not None else []



        
M = np.array([[0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 6, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 6, 5, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 5, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 9, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


# criação do grafo do MDP
estados = [State() for _ in range(len(M))]

for i in range(len(M)):
    
    for j in range(len(M)):
        if M[i,j] != 0: #existe caminho entre i e j
            estados[i].children.append(j)
            #children e uma lista dos indices dos estados aos quais o estado
            #i se conecta
            estados[i].parents.append(i)
            #parents e uma lista dos indices dos estados aos quais o estado
            #j esta conectado
            estados[i].cost.append(M[i,j])
            #cost e a lista dos custos de transicao do estado i para o
            #estado j

#for i in range(len(M)):
#    print(i)
#    print(estados[i].children)
#    print(estados[i].parents)
#    print(estados[i].cost)

V = np.concatenate((np.inf * np.ones(len(estados) - 1), [0]))
V2 = np.zeros(len(V))
tolerance = 1e-6
politica = np.zeros(len(estados))

while sum(abs(V - V2)) > tolerance:
    V2 = np.copy(V)  # Cópia de V
    
    for i in range(len(estados) - 2, -1, -1):
        
        C = []
        for j in range(len(estados[i].children)):
            C.append(estados[i].cost[j] + V[estados[i].children[j]])
        
        C = np.array(C) #conversão de C em um Numpy array para fazer uso das funções min e argmin
        V[i] = np.min(C)  # valor mínimo
        j_min = np.argmin(C)  # pega o índice do valor mínimo
        politica[i] = estados[i].children[j_min]

j = 0
U = []

while j != len(estados)-1:
    j = int(politica[j])
    U.append(j)#+1)

print(U)