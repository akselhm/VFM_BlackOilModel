# first visualize the data to: (I) gain information about the distribution for the PDF and (II) gain information about the regression function (linear, quadratic, ...)
# make the function for the MLE

import numpy as np
import pandas as pd
import csv      #not needed
from matplotlib import pyplot as plt
import seaborn as sns   #not yet sure if needed. Maybe to plot
from scipy import stats
from scipy.optimize import minimize 

from Pipe import Pipe


"""
# -- Make dataset containing both values at inlet and outlet--

# Make dataframes for values at inlet and outlet (have curently used dataset0 and dataset_noisy)
df_out = pd.read_csv ('data/dataset_noisy.csv', usecols= ["outlet pressure [Pa]", 
    "outlet oil flowrate [m3/s]",
    "outlet gas flowrate [m3/s]",
    "outlet water flowrate [m3/s]",
    "outlet holdup [-]"])
df_in = pd.read_csv ('data/dataset_noisy.csv', usecols= ["inlet pressure [Pa]", 
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
#print(df.head)


# Add void fraction as 1 - holdup
holdups = df.loc[:,"holdup [-]"]
holdups = holdups.to_numpy()
void_fractions = 1- holdups[:]

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
"""
# ------ Class for establishing relationship between the estimated and the "real" void fraction

class linear_relationship:
    # class for establishing a linear relationship by using some correlation
    def __init__(self) -> None:
        pass

    # Use some correlation (or try several) to make a new column in the dataset named "void fraction (Corrrelation)" or something like that
    def flowrate2superficialvel(self, flowrates, D = 0.2):
        # Function for turning flow rates into superficial velocities for a single phase
        #inputs:
        #   flowrates = numpy array with flowrates
        #   D = diameter of pipe (find some way of implementing without having to initialize a new pipe)
        #returns:
        #   array of superficail velocities
        return 4*flowrates/(np.pi*D**2)

    def Hughmark_arr(self, oil_flowrates, water_flowrates, gas_flowrates):
        #TODO: state what this function is and does
        #return: array or float, prediction(s) for the void fraction based on the Hughmark correalation

        #turn flow rates into superficial velocities (not needed?)
        j_o = self.flowrate2superficialvel(oil_flowrates)
        j_w = self.flowrate2superficialvel(water_flowrates)
        j_g = self.flowrate2superficialvel(gas_flowrates)

        #define last superficial velocities
        j_l = j_o + j_w
        j   = j_l + j_g

        pred_void_fractions = j_g/(1.2*j)
        return pred_void_fractions


    #TODO: add pred_void_fractions to dataframe (or make new with only void_fractions and pred_void_fractions?)

    #Examine the relationship between the predicted value of the void fraction from data Y compared to the actual void fraction given from data Y
    def plot_void_fractions(self, Hughmark_void_fractions, void_fractions): #void_fractions as input as well??
        #function for making a scatter plot visualising the relationship between the estimated void fraction from a correlation and the void fraction obtained from simulations in OLGA
        fig = plt.figure()
        plt.scatter(Hughmark_void_fractions, void_fractions, c ="blue", marker="x")
        plt.title("Measured vs predicted void fractions based on flow rates and pressure")
        plt.ylabel('Actual void fractions (from OLGA)')
        plt.xlabel('Predicted void fractions using the Hughmark correlation')
        fig.savefig('results/HugmarkVSvoidfracs.png')



# ----------- Class of MLE predictor -------------

class MLE_predictor:
    #class for estimating the maximum likelihood of the void fraction
    def __init__(self, relation, void_fractions, x_void_fractions):
        self.relation = relation #object for calculation a relationship between correlated and estimated void fraction
        self.void_fractions = void_fractions # "Real" void fractions estimated from OLGA
        self.x_void_fractions = x_void_fractions #void fraction that is estimated from some corelation 


    #MLE function to use in the optimization algorithm to "train" the estimator
    def MLE(self, parameters): #x_array is the void fractions predicted from correlations
        # Extract parameters
        const, beta, std_dev = parameters 
        # Predict the output
        pred = const + beta*self.x_void_fractions #make a prediction of the acttual void_fraction based on the estimated void fractions from some correlation
        # Calculate the log-likelihood for given distribution (most likely multivariate normal distribution)
        LL = np.sum(stats.norm.logpdf(self.void_fractions, pred, std_dev))
        # Negative log-likelihood (to minimize)
        neg_LL = -1*LL
        return neg_LL

    # Initialize a guess of the parameters: (const, beta, std_dev)
    ### arr = np.array([0.05, 1.0, 0.02]) #NOTE: extremely sensitive to first guess     #PUT OUTSIDE CLASS

    # Minimize the negative log-likelihood
    def make_mlemodel(self, arr):
        #input: numpyarray of intial guess: np.array(const, beta, std_dev)
        mlemodel = minimize(self.MLE, arr, method='L-BFGS-B') #try different methods (NOTE: remember to import minimize in main.py as well)
        return mlemodel

    ### mlemodel = make_mlemodel(arr)       #PUT OUTSIDE CLASS

    #Function for giving outputs from pressure and flowrates
    def make_MLE_prediction(self, oil_flowrate, water_flowrate, gas_flowrate, mlemodel):
        #obtain list of parameters [const, beta, std_dev]
        const, beta, std_dev = mlemodel.x 
        #make an initial prediction using the hughmark correlation
        hughmark_pred = self.relation.Hughmark_arr(oil_flowrate, water_flowrate, gas_flowrate)
        #make the final prediction for 
        prediction = const + hughmark_pred*beta
        return prediction


# ---------- TEST ------------------------
"""
# Make object of the MLE_predictor class 

relation = linear_relationship()

Hughmark_void_fractions = relation.Hughmark_arr(oil_flowrates, water_flowrates, gas_flowrates)

predictor = MLE_predictor(relation, void_fraction, Hughmark_void_fractions)


# Initialize a guess of the parameters: (const, beta, std_dev)
arr = np.array([0.05, 1.0, 0.02]) #NOTE: extremely sensitive to first guess 

# make optimizer object
mlemodel = predictor.make_mlemodel(arr)    

# -- check results for the optimization --
print("MLE converged: ", mlemodel.success)
print("parameters are: ", mlemodel.x)


# -- test on generated testset -- 
int_to_test = 2
df_test_out = pd.read_csv ('data/testset.csv', usecols= ["outlet pressure [Pa]", 
    "outlet oil flowrate [m3/s]",
    "outlet gas flowrate [m3/s]",
    "outlet water flowrate [m3/s]",
    "outlet holdup [-]"])

oil_flowrates_test = df_test_out.loc[:,"outlet oil flowrate [m3/s]"]
water_flowrates_test = df_test_out.loc[:,"outlet water flowrate [m3/s]"]
gas_flowrates_test = df_test_out.loc[:,"outlet gas flowrate [m3/s]"]

# Add void fraction as 1 - holdup
holdups_test = df_test_out.loc[:,"outlet holdup [-]"]
holdups_test = holdups_test.to_numpy()
void_fractions_test = 1- holdups_test[:]


#estimate flowrate
estimate = predictor.make_MLE_prediction(oil_flowrates_test[int_to_test], water_flowrates_test[int_to_test], gas_flowrates_test[int_to_test], mlemodel)

print("hughmark:", relation.Hughmark_arr(oil_flowrates_test[int_to_test], water_flowrates_test[int_to_test], gas_flowrates_test[int_to_test]))
print("predicted voidfrac:", estimate)
print("real voidfrac:",void_fractions_test[int_to_test])
"""


