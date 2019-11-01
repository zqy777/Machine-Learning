import numpy as np 
import matplotlib.pyplot as plt

p = [1, 2, 3, 4, 5]
accuracy = [0.948435, 0.983425, 0.984653, 0.977287, 0.965623]
plt.figure()
plt.plot(p, accuracy, color='g', linestyle='-')
plt.xlim((1, 5))
plt.xticks(np.linspace(1, 5, 5))
plt.yticks(accuracy)
plt.xlabel("P-value")
plt.ylabel("Best Accuracy for Validation")
plt.title("Best Validation Accuracy versus P-value")
plt.savefig("part3plot.png")
plt.show()
