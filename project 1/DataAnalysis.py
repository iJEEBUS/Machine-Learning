"""Data Analysis Preliminaries

A simple data visualization exercise.
Given a dataset of book sales per hour on Amazon perform the following:
  1 - Clean the data
  2 - Create a scatter plot of the data
  3 - Generate line of best fit
  4 - Plot line of best fit on data

Todo:
  1 - Generate a line of best fit using the formula given.
  2 - Plot line of best fit on graph along with data points.

@author Ron Rounsifer
@version 01.09.2019 (01.09.2019)
"""
import numpy as np
import matplotlib.pyplot as plt

# matplotlib customizations
plt.style.use('bmh')
plt.title('Books Sold Per Hour on Amazon')
plt.xlabel('Hour')
plt.ylabel('# of Books Sold')

def graph(f, steps, LOBF_str):
  """Graph line of best fit

  Passed a lambda function of the line of best fit (LOBF) and the numbers
  of instances to plot for the LOBF, each number is ran through the lambda
  function, where the output is finally plotted.

  args:
    f - lamba function that is the equation to be plotted
    steps - the values of x used
  """
  x = np.array(steps)
  y = f(x)
  plt.plot(x,y,linewidth=0.85, color='blue', label=LOBF_str)
  plt.legend(loc='upper left',  fontsize='medium')
  plt.show()

# Load data and remove nan data
data = np.loadtxt(open("data.txt", "rb"), delimiter=",")
rows_mask = ~np.any(np.isnan(data), axis=1)
data = data[rows_mask]

# Assign axis variables and add to plot
x = [time[0] for time in data]
y = [sales[1] for sales in data]
plt.scatter(x, y, s=0.85, marker='o', color='black')

# Calculating LOBF formula
n = len(data) # no. of obs
x_sum = sum(x)
y_sum = sum(y)
xy_sum = sum([a*b for a,b in zip(x,y)])
x_squared_sum = sum([i**2 for i in x])

numerator = (n * xy_sum) - (x_sum * y_sum)
denominator = (n * x_squared_sum) - (x_sum**2)
slope = numerator / denominator
intercept = (y_sum - (slope * x_sum)) / n

LOBF = 'y=' + str(round(intercept,2)) + '+' + str(round(slope,2)) + 'x'  

#graph(lambda x : intercept + (slope * x), range(0,n), LOBF)
