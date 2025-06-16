import numpy as np
from random import randint

class State:
    def __init__(self, index, reward=None, parents=None, children=None):
        self.reward = reward if reward is not None else []
        self.parents = parents if parents is not None else []
        self.children = children if children is not None else []
        self.index = index

class Environment:
    
    def __init__(self, AMatrix, run_DP = False):               #AMatrix é a matriz de associação
        
        self.n_states = len(AMatrix)
        n_states = self.n_states
        self.states = [State(i) for i in range(n_states)]
        
        for i in range(n_states):
            
            for j in range(n_states):
                if AMatrix[i,j] != 0:                          #existe caminho entre i e j
                    self.states[i].children.append(j)          #children e uma lista dos indices dos estados aos quais o estado i se conecta
                    self.states[i].parents.append(i)           #parents e uma lista dos indices dos estados aos quais o estado j esta conectado
                    self.states[i].reward.append(AMatrix[i,j]) #reward e a lista dos custos de transicao do estado i para o estado j
        
        for i in range(n_states):
            if len(self.states[i].children) == 0:
                self.states[i].children.append(-1)
            else:
                if len(self.states[i].children) == 1:
                    self.states[i].children.append(-1)
                    self.states[i].children.append(-1)
        
        self.current_state = self.states[0]
        self.final_state = self.states[n_states-1]
        self.terminated = False
        
        if run_DP == True:
            self.solve()
        else:
            self.optimal_policy = "optimal policy unknown"
            self.optimal_path = "optimal path unknown"
    
    def solve(self):
        states = self.states
        V = np.concatenate(((-np.inf) * np.ones(len(states) - 1), [0]))
        V2 = np.zeros(len(V))
        tolerance = 1e-6
        policy = np.zeros(len(states))

        while sum(abs(V - V2)) > tolerance:
            
            V2 = np.copy(V)  # Cópia de V
            
            for i in range(len(states) - 2, -1, -1):
                
                C = []
                for j in range(len(states[i].children)):
                    if self.states[i].children[j] != -1:
                        C.append(states[i].reward[j] + V[states[i].children[j]])
                
                C = np.array(C) #conversão de C em um Numpy array para fazer uso das funções max e argmax
                V[i] = np.max(C)  # valor máximo
                j_max = np.argmax(C)  # pega o índice do valor máximo
                policy[i] = j_max
                
                j = 0
                path = []
        
        self.Valuefunc = V
        
        while j != len(states)-1:
            j = states[j].children[int(policy[j])]
            path.append(j)      
        self.optimal_policy = policy
        self.optimal_path = path
    
    def step(self, action):
        if self.terminated == False:
            if self.current_state.children[action] == -1:
                reward = 0
                observation = self.current_state.index
            else:
                reward = self.current_state.reward[action]
                observation = self.current_state.children[action]
                self.current_state = self.states[observation] #non stochastic MDP
                
            if self.current_state == self.final_state:
                self.terminated = True
            
            state_array = np.zeros(self.n_states)
            state_array[self.current_state.index] = 1
            return state_array, reward, self.terminated
        else:
            print('Terminal point reached')
            state_array = np.zeros(self.n_states)
            state_array[15] = 1
            return state_array, 0.0, True
        
        
    
    def reset(self):
        self.current_state = self.states[0]
        self.terminated = False
        state_array = np.zeros(self.n_states)
        state_array[0] = 1
        return state_array
        
    def sample_action(self):
        random_action = np.int64(randint(0,1))
        return random_action