import numpy as np
import matplotlib.pyplot as plt
fileLocation = "/home/soon/Desktop/runs/"
xLabel = 'Exploration Number'
yLabel = 'Cumulative Instantaneous Regret'
x1 = np.load(fileLocation + "x1.npy")
y1s = np.load(fileLocation + "y1s.npy")
y1 = np.mean(y1s, axis = 0)
x2 = np.load(fileLocation + "x2.npy")
y2s = np.load(fileLocation + "y2s.npy")
y2 = np.mean(y2s, axis = 0)
x3 = np.load(fileLocation + "x3.npy")
y3s = np.load(fileLocation + "y3s.npy")
y3 = np.mean(y3s, axis = 0)
x4 = np.load(fileLocation + "x4.npy")
y4s = np.load(fileLocation + "y4s.npy")
y4 = np.mean(y4s, axis = 0)
x5 = np.load(fileLocation + "x5.npy")
y5s = np.load(fileLocation + "y5s.npy")
y5 = np.mean(y5s, axis = 0)
x6 = np.load(fileLocation + "x6.npy")
y6s = np.load(fileLocation + "y6s.npy")
y6 = np.mean(y6s, axis = 0)
x7 = np.load(fileLocation + "x7.npy")
y7s = np.load(fileLocation + "y7s.npy")
y7 = np.mean(y7s, axis = 0)
x8 = np.load(fileLocation + "x8.npy")
y8s = np.load(fileLocation + "y8s.npy")
y8 = np.mean(y8s, axis = 0)
x9 = np.load(fileLocation + "x9.npy")
y9s = np.load(fileLocation + "y9s.npy")
y9 = np.mean(y9s, axis = 0)
modelString = "All Models"
modelString1 = "PMF_Boltzmann"
modelString2 = "PMF_Exploit"
modelString3 = "PMF_UCB"
modelString4 = "PMF_Thompson_Sampling"
modelString5 = "PMF_Entropy"
modelString6 = "PMF_eGreedy"
modelString7 = "Optimal"
modelString8 = "Worst"
modelString9 = "PMF_Random"
plt.plot(x1, y1, label=modelString1)
plt.plot(x2, y2, label=modelString2)
plt.plot(x3, y3, label=modelString3)
plt.plot(x4, y4, label=modelString4)
plt.plot(x5, y5, label=modelString5)
plt.plot(x6, y6, label=modelString6)
plt.plot(x7, y7, label=modelString7)
plt.plot(x8, y8, label=modelString8)
plt.plot(x9, y9, label=modelString9)
plt.legend(loc = 'upper left')
plt.xlabel(xLabel)
plt.ylabel(yLabel)
plt.title(modelString)
plt.savefig(fileLocation + "AllInOneReloaded.png")
plt.clf()
