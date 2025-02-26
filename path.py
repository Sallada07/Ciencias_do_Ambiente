import numpy as np
import matplotlib.pyplot as plt

def gerar_zigue_zague(pontos):
    # Ordenando os pontos (em caso de não estarem na ordem certa)
    pontos = sorted(pontos, key=lambda p: (p[0], p[1]))

    # Determinando as coordenadas do retângulo
    x_min = min(p[0] for p in pontos)
    x_max = max(p[0] for p in pontos)
    y_min = min(p[1] for p in pontos)
    y_max = max(p[1] for p in pontos)

    # Gerar o caminho em zigue-zague contínuo dentro do retângulo
    caminho = []
    for y in np.arange(y_min, y_max, 1):  # Passo de 1 para o movimento em y
        if int(y) % 2 == 0:
            # Movimento da esquerda para a direita
            caminho.append((x_min, y))
            caminho.append((x_max, y))
        else:
            # Movimento da direita para a esquerda
            caminho.append((x_max, y))
            caminho.append((x_min, y))

    return caminho, (x_min, x_max, y_min, y_max)

def plotar_caminho(caminho, limites):
    # Desenhar o retângulo
    plt.figure(figsize=(8, 6))
    x_min, x_max, y_min, y_max = limites
    
    # Plotar o caminho em zigue-zague (conectando os pontos)
    x_vals, y_vals = zip(*caminho)
    plt.plot(x_vals, y_vals, 'b-', lw=1)
    
    # Configurar os limites do gráfico
    plt.xlim(x_min - 1, x_max + 1)
    plt.ylim(y_min - 1, y_max + 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Caminho Zigue-Zague")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

# Exemplo de pontos do retângulo (4 pontos: [x, y])
pontos = [(0, 0), (0, 50), (50, 0), (50, 50)]

# Gerar o caminho em zigue-zague
caminho, limites = gerar_zigue_zague(pontos)

# Plotar o gráfico com o caminho
plotar_caminho(caminho, limites)
