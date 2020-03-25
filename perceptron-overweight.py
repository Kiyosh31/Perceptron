import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self, dim, eta):
        self.n = dim
        self.eta = eta
        # pesos sinapticos
        self.w = -1 + 2 * np.random.random((dim, 1))
        # sesgo o bias
        self.b = -1 + 2 * np.random.random()
        
    def predict(self, x):
        # producto punto
        y = np.dot(self.w.transpose(), x) + self.b
        if y > 25.0:
            return 1
        else:
            return 0
        
    def train(self, X, y, epochs = 50):
        # X -> entradas
        # Y -> valores deseados
        n, m = X.shape
        for i in range(epochs):
            for j in range(m):
                y_pred = self.predict(X[:,j])
                self.w += self.eta*(y[j] - y_pred) * X[:,j].reshape(-1,1)
                self.b += self.eta*(y[j] - y_pred)



def normalize(x):
  return [(x_i - np.mean(x_i))/np.std(x) for x_i in x]



samples = 50
heights = 50 + (200 - 50) * np.random.rand(samples) 
weights = 30.0 + (150.0 - 30.0) * np.random.rand(samples)

X = np.array((weights.ravel(), heights.ravel()))
# print('Datos Personas \n', X)

imc = weights/((heights/100)**2)
imc = imc.ravel()
y = list()

for i in imc:
    if(i > 25.0):
        y.append(1)
    else:
        y.append(0)

y = np.array(y)
# print('Salidas esperadas \n', y)


# Creacion y entrenamiento de red
net = Neuron(2, 0.5)
net.train(X, y, 100)
w = net.w
b = net.b

# Valores Finales (datos sin escalar)
print('Ejecucion 1')
plt.figure()
plt.grid()
plt.plot(weights[y == 0], heights[y == 0], 'g^')
plt.plot(weights[y == 1], heights[y == 1], 'ro')
plt.xlim(0, 220)
plt.ylim(0, 220)
plt.plot([0,220],[(-w[0]/w[1])*(0)-(b/w[1]),(-w[0]/w[1])*(220)-(b/w[0])],'-m')
plt.xlabel("Peso (Kg)")
plt.ylabel("Altura (Cm)")
plt.title("Resultado datos sin escalar")
plt.show()

# Valores Finales (datos escalados)
X = np.array(normalize(X))

net.train(X, y, 100)
w = net.w
b = net.b

plt.title("Resultado datos escalados")
plt.grid()
plt.plot(X[0][y == 0], X[1][y == 0], '*b')
plt.plot(X[0][y == 1], X[1][y == 1], '.r')
plt.plot([-2,2],[(-w[0]/w[1])*(-2)-(b/w[1]),(-w[0]/w[1])*(2)-(b/w[1])],'-m')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.xlabel("Peso")
plt.ylabel("Altura")
plt.show()

# Segunda ejecucion (datos escalados)
heights_1 = 50 + (200 - 50) * np.random.rand(samples) 
weights_1 = 30.0 + (150.0 - 30.0) * np.random.rand(samples)
X_1 = np.array((weights_1.ravel(), heights_1.ravel()))
X_1 = np.array(normalize(X_1))

net.train(X, y, 100)
w = net.w
b = net.b

print('Ejecucion 2')
plt.title("Resultado datos escalados")
plt.grid()
plt.plot(X[0][y == 0], X[1][y == 0], '*b')
plt.plot(X[0][y == 1], X[1][y == 1], '.r')
plt.plot([-2,2],[(-w[0]/w[1])*(-2)-(b/w[1]),(-w[0]/w[1])*(2)-(b/w[1])],'-m')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.xlabel("Peso")
plt.ylabel("Altura")
plt.show()