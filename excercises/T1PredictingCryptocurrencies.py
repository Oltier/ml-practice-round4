import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read the data
df = pd.read_csv('BTC_ETH_round4.csv')
x = df.Bitcoin.values  # Bitcoin values
y = df.Ethereum.values  # Ethereum values

# Reshape the data.
# This is really important for the matrix multiplications later on!
x = np.reshape(x, (len(x), 1))
y = np.reshape(y, (len(y), 1))

# Plot the data
plt.figure(figsize=(6, 6))
plt.scatter(x, y)
plt.title(r'$\bf{Figure\ 1.}$ Bitcoin vs Ethereum')
plt.xlabel('Bitcoin')
plt.ylabel('Ethereum')
plt.show()
print(len(x))
