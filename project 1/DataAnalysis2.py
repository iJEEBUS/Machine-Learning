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

class DataAnalysis(object):
    
  def init(self):
    """ Constructor
    
    Create an empty data set.
    """
    self.data = None
    
  def load(self, filename, **kwargs):
    """Load data
    
    Loads the file specified by the users and applies filters if passed.
    """
    # Load data
    self.data = np.loadtxt(open(filename, 'rb'), delimiter=',')
    
    # check for keywords
    if len(kwargs) > 0:
      
      # filter the data
      if 'filters' in kwargs.keys():
        if 'delete' == kwargs['filters']:
          data_filter = ~np.any(np.isnan(self.data), axis=1)
          self.data = self.data[data_filter]
        else:
          print("The filter you entered is not valid. No filter applied.\n Check load() method.")

  def SLR(self):
    # Assign axis variables and add to plot
    x = [time[0] for time in self.data]
    y = [sales[1] for sales in self.data]
    plt.scatter(x, y, s=0.85, marker='o', color='black')

    # Calculating LOBF formula
    num_obs = len(self.data) 
    x_sum = sum(x)
    y_sum = sum(y)
    xy_sum = sum([a*b for a,b in zip(x,y)])
    x_squared_sum = sum([i**2 for i in x])

    numerator = (num_obs * xy_sum) - (x_sum * y_sum)
    denominator = (num_obs * x_squared_sum) - (x_sum**2)
    slope = numerator / denominator
    intercept = (y_sum - (slope * x_sum)) / num_obs
    
    # String version for the legend
    LOBF = 'y=' + str(round(intercept,2)) + '+' + str(round(slope,2)) + 'x'  
    return num_obs, intercept, slope, LOBF

  def QR(self):
    # Assign axis variables and add to plot
    x = [time[0] for time in self.data]
    x_squared = [x**2 for x in x]
    y = [sales[1] for sales in self.data]
    plt.scatter(x, y, s=0.85, marker='o', color='black')
    
    # calculate line of best fit
    num_obs = len(self.data)
    x_sum = sum(x)
    x_squared_sum = sum(x_squared)
    y_sum = sum(y)
    mean_y = sum(y)/num_obs
    mean_x = sum(x)/num_obs
    mean_x_squared = sum(x_squared)/num_obs
    
    y_x_sum = sum([y*x for y,x in zip(y,x)])
    mean_y_x = (y_sum*x_sum)/num_obs
    s_y1 = y_x_sum - mean_y_x 
    
    s_22 = sum([x**2 for x in x_squared]) - ((x_squared_sum**2)/num_obs)
    
    y_x_squared_sum = sum([y*x for y,x in zip(y,x_squared)])
    s_y2 =  y_x_squared_sum - ((y_sum * x_squared_sum)/num_obs)  

    s_12 = sum([x*x_squared for x,x_squared in zip(x, x_squared)]) - ((x_sum*x_squared_sum)/num_obs)
    s_11 = sum([x**2 for x in x]) - (x_sum**2/num_obs)

    beta_two = ((s_y1*s_22)-(s_y2*s_12)) / ((s_22*s_11)-(s_12**2))
    beta_three = ((s_y2*s_11)-(s_y1*s_12)) / ((s_22*s_11)-(s_12**2))  
    beta_one  = mean_y - (beta_two*mean_x) - (beta_three*mean_x_squared)
    
    LOBF = 'y=' + str(round(beta_one,2)) + '+' + str(round(beta_two,2)) + 'x+' + str(round(beta_three,2)) + 'x^2'   

    return num_obs, beta_one, beta_two, beta_three, LOBF
    # y = b1 + b2x1 + b3x2

  def graph_multiple_functions(self, funcs, steps, LOBF_strs):
    colors = ['blue', 'red', 'cyan']

    x = np.array(steps)
    for i in range(len(funcs)-1):
      y = funcs[i](x)
      plt.plot(x,y,line_width=0.85, color=colors[i], label=LOBF_strs[i])
      plt.legend(loc='best', fontsize='medium')
    plt.show()
   
  def graph(self, f, steps, LOBF_str):
    """Graph line of best fit

    Passed a lambda function of the line of best fit (LOBF) and the numbers
    of instances to plot for the LOBF, each number is ran through the lambda
    function, where the output is finally plotted.

    args:
      f - lamba function that is the equation to be plotted
      steps - the values of x used
    """
    
    # matplotlib customizations
    plt.style.use('bmh')
    plt.title('Books Sold Per Hour on Amazon')
    plt.xlabel('Hour')
    plt.ylabel('# of Books Sold')

    x = np.array(steps)
    y = f(x)
    plt.plot(x,y,linewidth=0.85, color='blue', label=LOBF_str)
    plt.legend(loc='upper left',  fontsize='medium')
    plt.show()


data = DataAnalysis()
data.load('./data.txt', filters='delete')

# SLR
num_obs, intercept, slope, LOBF_slr = data.SLR()
#data.graph(lambda x : intercept + (slope*x), range(0, num_obs), LOBF_slr)

# QR
num_obs, b1, b2, b3, LOBF_qr = data.QR()
#data.graph(lambda x : b1 + (b2*x) + (b3*(x**2)), range(0,num_obs), LOBF_qr)

# Graph both on same plot
y_slr = [intercept + (slope*x) for x in data.x] #TODO this will throw an error!!!
y_qr =
functions = [lambda x : intercept + (slope*x), lambda x : b1 + (b2*x) + (b3*(x**2))] 
labels = [LOBF_slr, LOBF_qr]
data.graph_multiple_functions(functions, range(0, num_obs), labels)

# can you convert the outputs of the lambdas into a numpy array and then 
# pass those values of x to the graphing function?
