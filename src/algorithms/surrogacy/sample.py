import pandas as pd

import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("/home/thivi/OpenRASE/src/algorithms/surrogacy/data/weights.csv", sep=r"\s*,\s*", engine="python")

# Plot the data
# plt.figure(figsize=(10, 6))
# plt.plot(data['reqps'], data['latency'], marker='o', linestyle='-', color='b')
# plt.xlabel('CPU')
# plt.ylabel('Latency')
# plt.title('CPU vs Latency')
# plt.grid(True)
# plt.savefig('/home/thivi/OpenRASE/src/algorithms/surrogacy/data/weights.png')

plt.figure(figsize=(10, 6))
plt.boxplot(data["latency"])
plt.savefig('/home/thivi/OpenRASE/src/algorithms/surrogacy/data/weights.png')

q1 = data["latency"].quantile(0.25)
q3 = data["latency"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

filteredData = data.head(1462)[(data["latency"] > lower_bound) & (data["latency"] < upper_bound) & (data["sfc"] == "sfc2-1")]
filteredData1 = data.head(1462)[(data["latency"] > lower_bound) & (data["latency"] < upper_bound) & (data["sfc"] == "sfc0-0")]

plt.figure(figsize=(10, 6))
plt.subplot(2, 1,1 )
plt.plot(range(1, len(filteredData.index)+1), filteredData["latency"], marker="x", color="y")
plt.plot(range(1, len(filteredData1.index)+1), filteredData1["latency"], marker="x", color="r")
plt.subplot(2,1,2)
plt.plot(range(1, len(filteredData.index)+1), filteredData["reqps"], marker="x", color="y")
plt.plot(range(1, len(filteredData1.index)+1), filteredData1["reqps"], marker="x", color="r")


plt.xlabel('CPU')
plt.ylabel('Latency')
plt.title('CPU vs Latency')
plt.grid(True)
plt.savefig('/home/thivi/OpenRASE/src/algorithms/surrogacy/data/weights.png')
