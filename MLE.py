# first visualize the data to: (I) gain information about the distribution for the PDF and (II) gain information about the regression function (linear, quadratic, ...)
# make the function for the MLE

import numpy as np
import pandas as pd
import csv
from matplotlib import pyplot as plt
import seaborn as sns   #not yet sure if needed. Maybe to plot
from scipy import stats
from scipy.optimize import minimize 

from Pipe import Pipe

#TODO: handle dataset: split, etc.. (not sure that initially should be done here or somewhere else, like in main)

#TODO: visualize data. plot void fraction as a function of pressure and as a function of oil flow rate

"""
df = pd.read_csv ('data/dataset1.csv', usecols= ['outlet oil flowrate [Sm3/s]','outlet holdup [-]'])

print(df.shape)
print(type(df))
#selct only valid simulations (here the first 349)
df = df.head(349)
print(df.shape)

#add void fraction as 1 - holdup
holdups = df.loc[:,"outlet holdup [-]"]
holdups = holdups.to_numpy()
void_fractions = 1- holdups[:]

#print(type(void_fractions))
#print(void_fractions.shape)

exit()

sns.regplot(x='outlet oil flowrate [Sm3/s]', y='outlet holdup [-]', data = df)
plt.savefig('results/flowrate_holdup.png')

array = df.to_numpy()
print(type(array))
#print(df[1][:])
"""

# -- Make dataset containing both values at inlet and outlet--
#M ake dataframes for values at inlet and outlet 
df_out = pd.read_csv ('data/dataset0.csv', usecols= ["outlet pressure [Pa]", 
    "outlet oil flowrate [m3/s]",
    "outlet gas flowrate [m3/s]",
    "outlet water flowrate [m3/s]",
    "outlet holdup [-]"])
df_in = pd.read_csv ('data/dataset0.csv', usecols= ["inlet pressure [Pa]", 
    "inlet oil flowrate [m3/s]", 
    "inlet gas flowrate [m3/s]", 
    "inlet water flowrate [m3/s]", 
    "inlet holdup [-]"])

# Rename columns
df_out.columns = ["pressure [Pa]", 
    "oil flowrate [m3/s]", 
    "gas flowrate [m3/s]", 
    "water flowrate [m3/s]", 
    "holdup [-]"]
df_in.columns = ["pressure [Pa]", 
    "oil flowrate [m3/s]", 
    "gas flowrate [m3/s]", 
    "water flowrate [m3/s]", 
    "holdup [-]"]

# Concatenate dataframes
df = pd.concat([df_in, df_out])
print(df.head)

# Add void fraction as 1 - holdup
holdups = df.loc[:,"holdup [-]"]
holdups = holdups.to_numpy()
void_fractions = 1- holdups[:]
#TODO: add to df

# -- Make a prediction of the void fraction based on the parameters (P, q_i, j_i, ..?) --
# Retrieve pressures and flow rates
pressures = df.loc[:,"pressure [Pa]"]
oil_flowrates = df.loc[:,"oil flowrate [m3/s]"]
water_flowrates = df.loc[:,"water flowrate [m3/s]"]
gas_flowrates = df.loc[:,"gas flowrate [m3/s]"]

# Turn to numpy array
pressures = pressures.to_numpy()
oil_flowrates = oil_flowrates.to_numpy()
water_flowrates = water_flowrates.to_numpy()
gas_flowrates = gas_flowrates.to_numpy()

# Use some correlation (or try several) to make a new column in the dataset named "void fraction (Corrrelation)" or something like that

def flowrate2superficialvel(flowrates, D = 0.2):
    # Function for turning flow rates into superficial velocities for a single phase
    #inputs:
    #   flowrates = numpy array with flowrates
    #   D = diameter of pipe (find some way of implementing without having to initialize a new pipe)
    #returns:
    #   array of superficail velocities
    return 4*flowrates/(np.pi*D**2)

def Hughmark_arr(oil_flowrates, water_flowrates, gas_flowrates):
    #TODO: state what this function is and does
    #return: prediction for the void fraction

    #turn flow rates into superficial velocities
    j_o = flowrate2superficialvel(oil_flowrates)
    j_w = flowrate2superficialvel(water_flowrates)
    j_g = flowrate2superficialvel(gas_flowrates)

    #define last superficial velocities
    j_l = j_o + j_w
    j   = j_l + j_g

    pred_void_fractions = j_g/(1.2*j)
    return pred_void_fractions

Hughmark_void_fractions = Hughmark_arr(oil_flowrates, water_flowrates, gas_flowrates)

#TODO: add pred_void_fractions to dataframe (or make new with only void_fractions and pred_void_fractions?)

"""
Fr_j = j/(np.sqrt(9.81*self.pipe.D))
if Fr_j < 3.5:
    Cd = 1.05 #...+ 0.15*np.sin(angle)
    Ud = np.sqrt(9.81*self.pipe.D)*0.54 #np.sqrt(g_s*D)*(0.35*np.sin(angle) + 0.54*np.cos(angle))
else: #Fr_j >= 3.5
    Cd = 1.2
    Ud = 0 #0.35*np.sqrt(9.81*self.pipe.D)*np.sin(angle)
return Cd, Ud
"""


#TODO: Examine the relationship between the predicted value of the void fraction from data Y compared to the actual void fraction given from data Y

#make a plot similar to the one above, only that this hopefully will provide more information
plt.scatter(Hughmark_void_fractions, void_fractions, c ="blue", marker="x")
plt.savefig('results/HugmarkVSvoidfracs.png')

exit()

#TODO: define an MLE function containing: (I) the model building equation and (II) the logarithmic value of the probability density function

y = 1 #data array (initially set to 1)
#Sceleton for MLE-function
def MLE(parameters):
    #extract parameters
    const, beta1, beta2, V = parameters #not sure if this works with a covariate matrix V (also, what is the difference between V_0 and V?)
    #predict the output
    pred = const #+++
    #calculate the log-likelihood for given distribution (most likely multivariate normal distribution)
    LL = np.sum(stats.multivariate_normal.logpdf(y, pred, V))
    #negative log-likelihood (to minimize)
    neg_LL = -1*LL
    return neg_LL

MLE_function = MLE(y)
#TODO: minimize the negative log-likelihood

arr= np.array([0,0])
mlemodel = minimize(MLE_function, arr, method=...)