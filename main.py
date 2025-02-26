import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from   pyswarm import pso
from   random import random

class Signal:
    def __init__(self, id=0, localization:tuple=(0,0)) -> None:
        self.x, self.y = localization
        self.id = id

    def send_signals(self) -> list:
        return [Signal(id=self.id, localization=(self.x, self.y)) for _ in range(5)]
    
    def recive_signals(self, signals:list):
        xm = sum([signal.x-self.x for signal in signals])/len(signals)
        ym = sum([signal.y-self.y for signal in signals])/len(signals)
        d = (xm**2 + ym**2)**(1/2)
        return d
    

class Anchor(Signal):
    def solicita_sinais(self, ancs:list, num_obj:int) -> list:
        dists = [0]*num_obj
        for anc in ancs:
            d = self.recive_signals(anc.send_signals())
            dists[anc.id] = d
        return dists
        

class Robo(Anchor):
    def __init__(self, id, localization: tuple):
        super().__init__(id, localization)
        self.ancs = []
        self.kalman_state = np.array([[self.x], [self.y]], dtype=np.float64)  # Posição inicial (x, y)
        self.kalman_cov = np.eye(2) * 10 # Covariância inicial
        self.process_noise = np.eye(2) * 0.1  # Ruído do processo
        self.measurement_noise = np.eye(2) * 3  # Ruído da medição

    def cria_mapa(self, ancs:list[Anchor]) -> None:
        # Automação para vários pontos:
        num_obj = len(ancs)
        for i, anc in enumerate(ancs):
            # Demais pontos
            if i != 0 and i != 1: 
                d2 = max(ancs[0].solicita_sinais([anc], num_obj))
                d3 = max(ancs[1].solicita_sinais([anc], num_obj))
                x = (d1**2 + d2**2 - d3**2)/(2*d1)
                y = (d2**2 - x**2)**(1/2)
                self.ancs.append(Anchor(id = anc.id, localization = (x, y)))
            # Segundo ponto
            elif i == 1:
                dists_temp = ancs[0].solicita_sinais([anc], num_obj)
                d1 = max(dists_temp)
                self.ancs.append(Anchor(id = anc.id, localization = (d1, 0)))
            # Primeiro ponto
            else: 
                self.ancs = [Anchor(id = anc.id, localization = (0,0))]

        xs = [signal.x for signal in self.ancs]
        ys = [signal.y for signal in self.ancs]
        self.map_lim = [[min(xs), min(ys)], [max(xs), max(ys)]]
        return self.ancs

    def triangulate(self, A, C, B, dA, dB, dC):
        def erro(posicao, A, C, B, dA, dB, dC):
            x, y = posicao
            eq1 = np.sqrt((x - A[0])**2 + (y - A[1])**2) - dA
            eq2 = np.sqrt((x - B[0])**2 + (y - B[1])**2) - dB
            eq3 = np.sqrt((x - C[0])**2 + (y - C[1])**2) - dC
            return eq1**2 + eq2**2 + eq3**2  # Minimiza o erro quadrado

        # Definindo limites de busca para as coordenadas
        lb = self.map_lim[0]  # Limite inferior (para x e y)
        ub = self.map_lim[1]    # Limite superior (para x e y)
        
        # Usando PSO para otimizar a posição do ponto
        solution, _ = pso(erro, lb, ub, args=(A, C, B, dA, dB, dC))
        
        return solution

    def localizacao_atual(self) -> tuple:
        """ Estima a posição atual do robô via triangulação e aplica Filtro de Kalman. """

        # Triangulações
        points = []
        for anc in self.ancs:
            temp = self.ancs.copy()
            temp.remove(anc)
            dists = self.solicita_sinais(self.ancs, len(self.ancs))
            dists = [dists[anchor.id] for anchor in temp]
            temp = [(anchor.x, anchor.y) for anchor in temp] 
            points.append(self.triangulate(*temp, *dists))
        
        # média
        xm = sum([point[0] for point in points])/len(points)
        ym = sum([point[1] for point in points])/len(points)

        estimated_position = (xm, ym)

        # Aplicando o Filtro de Kalman
        H = np.eye(2)
        K = self.kalman_cov @ H.T @ np.linalg.inv(H @ self.kalman_cov @ H.T + self.measurement_noise)

        measurement = np.array([[estimated_position[0]], [estimated_position[1]]])
        self.kalman_state += K @ (measurement - H @ self.kalman_state)
        self.kalman_cov = (np.eye(2) - K @ H) @ self.kalman_cov + self.process_noise

        x,y = self.kalman_state.flatten()
        return x,y


# Inicializa a posição das ancoras
ancs = [Anchor(0, (0, 0)), Anchor(1, (0, 50)), Anchor(2, (50, 0)), Anchor(3, (50, 50))]
robo = Robo(4, (50*random(), 50*random()))
ancs_fake = robo.cria_mapa(ancs)
# Inicializa a posição do robo
x_data_real = [robo.x]  # Lista para armazenar coordenadas X
y_data_real = [robo.y]  # Lista para armazenar coordenadas Y
# Robo se localiza
robo.localizacao_atual()
x_data_fake = [robo.x]  # Lista para armazenar coordenadas X
y_data_fake = [robo.y]  # Lista para armazenar coordenadas Y

# Criação da figura
fig, ax = plt.subplots()
ax.set_xlim(-1, 51)
ax.set_ylim(-1, 51)

# Elementos gráficos
anchors = ax.scatter([anchor.x for anchor in ancs], [anchor.y for anchor in ancs], c='green', marker='o', label="Âncoras", s=50)
anchors_fake = ax.scatter([anchor.x for anchor in ancs_fake], [anchor.y for anchor in ancs_fake], c='m', marker='o', label="Âncoras", s=50)
point_real, = ax.plot([], [], 'ro', markersize=8, label="Robô_real")  # Ponto vermelho
path_real, = ax.plot([], [], 'b--', linewidth=1.5, label="Caminho_real")  # Linha tracejada azul
point_fake, = ax.plot([], [], 'yo', markersize=8, label="Robô_visão")  # Ponto vermelho
path_fake, = ax.plot([], [], 'g--', linewidth=1.5, label="Caminho_visão")  # Linha tracejada azul

for i in range(20):
    x_new = x_data_real[-1] + np.random.choice([-1, 0, 1])
    y_new = y_data_real[-1] + np.random.choice([-1, 0, 1])
    print(x_new, y_new)
    x_new, y_new = robo.localizacao_atual()
    print(x_new, y_new)
    print()












# # Função para atualizar a posição do ponto
# def update(frame):
#     global x_data_real, y_data_real, x_data_fake, y_data_fake

#     # Lógica para decidir a próxima posição (exemplo: movimento aleatório)
#     x_new = x_data_real[-1] + np.random.choice([-1, 0, 1])
#     y_new = y_data_real[-1] + np.random.choice([-1, 0, 1])
#     # Atualiza os vetores com a nova posição
#     print(x_new, y_new)
#     x_data_real.append(x_new)
#     y_data_real.append(y_new)


#     x_new, y_new = robo.localizacao_atual()
#     print(x_new, y_new)
#     # Atualiza os vetores com a nova posição
#     x_data_fake.append(x_new)
#     y_data_fake.append(y_new)
    
#     # Atualiza o gráfico
#     point_real.set_data(x_data_real[-1], y_data_real[-1])
#     path_real.set_data(x_data_real, y_data_real)  # Atualiza a trilha percorrida
#     point_fake.set_data(x_data_fake[-1], y_data_fake[-1])
#     path_fake.set_data(x_data_fake, y_data_fake)  # Atualiza a trilha percorrida
#     return point_real, path_real, point_fake, path_fake

# # Criação da animação
# ani = animation.FuncAnimation(fig, update, frames=100, interval=300, blit=False)
# plt.show()
