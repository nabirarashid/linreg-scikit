from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

#only accepts 2D arrays for x values; y values can be 1D
temperature = np.array(range(80,120,2))
#-1 means the number of rows is determined based on the total number of elements in the array
temperature = temperature.reshape(-1,1)
sales = [65, 58, 46, 45, 44, 42, 40, 40, 36, 38, 38, 28, 30, 22, 27, 25, 25, 20, 15, 5]

line_fitter = LinearRegression()
line_fitter.fit(temperature,sales)
predicted_sales = line_fitter.predict(temperature)

plt.plot(temperature,predicted_sales)
plt.plot(temperature, sales, 'o')
plt.show()
