import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from blackoil import black_oil
from Pipe import Pipe
from VFM import TH_model
from MLE import linear_relationship, MLE_predictor

#increse fonsize of plots
plt.rcParams.update({'font.size': 15})

#OPTIONS

run_mechanistic_model = True
run_datadriven_model = True

# -- initialize values --

# black oil
GOR = 50.0  #[-]
wc = 0.30   #[-]
p_bp = 50.0e5   #[Pa] 
t_bp = 20.0 + 273.15 # [K]
rho_o0 = 867.0  #[kg/m^3] 
rho_g0 = 0.997      #[kg/m^3]
rho_w0 = 1020.0     #[kg/m^3]
gamma_g0 = 0.814 #0.799    # [-] #could be calculated from rho_g0 and rho_air
T_r = 25 + 273.15    # [K] included in black oil as it is constant

# pipe
D = 0.2     # [m]
eps = 3e-5#0.00021#3e-5 # [m]
length = 1000   # [m]
N = 500 

# thermal-hydraulic model
q_l_in = 0.157726   # [sm^3/s] 
p_out = 10e5    # [Pa] 
#void_frac_method = "Bendiksen" 
twophase_ff_method = "VG"


# ------------ MLE object -----------------

# -- handle dataset for MLE --

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

# Obtain void fraction as 1 - holdup
holdups = df.loc[:,"holdup [-]"]
holdups = holdups.to_numpy()
void_fractions = 1- holdups[:]

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


# -- Make object of the MLE_predictor class --

relation = linear_relationship()

Hughmark_void_fractions = relation.Hughmark_arr(oil_flowrates, water_flowrates, gas_flowrates)

predictor = MLE_predictor(relation, void_fractions, Hughmark_void_fractions)


# Initialize a guess of the parameters: (const, beta, std_dev)
arr = np.array([0.05, 1.0, 0.025]) #NOTE: extremely sensitive to first guess 

# make optimizer object
mlemodel = predictor.make_mlemodel(arr)    

# -- check results for the optimization --
print("MLE converged: ", mlemodel.success)
print("parameters are: ", mlemodel.x)



# --------------------------------------------

# -- initialize objects: fluid, pipe, TH_model --
fluid = black_oil(GOR, wc, p_bp, t_bp, rho_o0, rho_g0, rho_w0, gamma_g0, T_r)
pipe = Pipe(D, eps, length, N)
model = TH_model(fluid, predictor, mlemodel, pipe, q_l_in, p_out)



# run algorithm for mechanistic approach
#B_pressures = model.Andreolli_algorithhm("Bendiksen", twophase_ff_method, print_iteration=True) 

#WG_pressures = model.Andreolli_algorithhm("WG", twophase_ff_method)

#MLE_pressures = model.Andreolli_algorithhm("MLE", twophase_ff_method)

# -- run algorithm on dataset and see results --

# get dataset with necessary BCs
dataset_p75 = pd.read_csv ('data/dataset_p75.csv', usecols= ["inlet pressure [Pa]",
    "outlet pressure [Pa]", 
    "inlet oil flowrate [m3/s]",
    "inlet water flowrate [m3/s]"
    ])

dataset_p125 = pd.read_csv ('data/dataset_p125.csv', usecols= ["inlet pressure [Pa]",
    "outlet pressure [Pa]", 
    "inlet oil flowrate [m3/s]",
    "inlet water flowrate [m3/s]"
    ])

def test_model(df, pressure):
    #Function for testing the mechanistic model on dataset "df" which has the constant outlet pressure "pressure", both given as parameters
    #----------------------------
    #inputs:_____________________
    #df: dataframe of testset
    #pressure: Constant outlet pressure given in Pascal 
    #----------------------------

    #sort in ascending inlet liquid/oil void fraction
    df = df.sort_values(by=["inlet oil flowrate [m3/s]"])

    pressures_in = df.loc[:,"inlet pressure [Pa]"].to_numpy()
    pressures_out = df.loc[:,"outlet pressure [Pa]"].to_numpy() #corresponds to the pressure at the second last step (right before the outlet)
    oil_flowrates_in = df.loc[:,"inlet oil flowrate [m3/s]"].to_numpy()
    water_flowrates_in = df.loc[:,"inlet water flowrate [m3/s]"].to_numpy()
    ql_in_array = oil_flowrates_in + water_flowrates_in
    #pressures_in = pressures_in.to_numpy()

    #make arrays with results
    B_inlet_pressures = np.zeros(len(pressures_in))
    WG_inlet_pressures = np.zeros(len(pressures_in))
    MLE_inlet_pressures = np.zeros(len(pressures_in))

    #errors
    B_error = np.zeros(len(pressures_in))
    WG_error = np.zeros(len(pressures_in))
    MLE_error = np.zeros(len(pressures_in))
    for i in range(len(pressures_in)):
        print("iteration at", i)
        #make new model with new input values
        model = TH_model(fluid, predictor, mlemodel, pipe, ql_in_array[i], pressure)

        #run model with Bendiksen
        B_pressures = model.Andreolli_algorithhm("Bendiksen", twophase_ff_method)
        B_inlet_pressures[i] = B_pressures[0] #add computed inlet pressure to result array
        B_error[i] = (pressures_in[i] - B_inlet_pressures[i])*100/pressures_in[i]# - pressure)

        #run model with Woldesmayat and Gajar
        WG_pressures = model.Andreolli_algorithhm("WG", twophase_ff_method)
        WG_inlet_pressures[i] = WG_pressures[0] #add computed inlet pressure to result array
        WG_error[i] = (pressures_in[i] - WG_inlet_pressures[i])*100/pressures_in[i]# - pressure)

        #run model with MLE for void fraction
        MLE_pressures = model.Andreolli_algorithhm("MLE", twophase_ff_method)
        MLE_inlet_pressures[i] = MLE_pressures[0] #add computed inlet pressure to result array
        MLE_error[i] = (pressures_in[i] - MLE_inlet_pressures[i])*100/pressures_in[i]# - pressure)

    #plot the inlet pressures
    fig = plt.figure()
    plt.plot(ql_in_array, pressures_in, 'k' , label="real (OLGA)")
    plt.plot(ql_in_array, B_inlet_pressures, 'b--' ,label="Bendiksen")
    plt.plot(ql_in_array, WG_inlet_pressures, 'r-.', label="W & G")
    plt.plot(ql_in_array, MLE_inlet_pressures, 'g--', label="MLE")
    #plt.title("Constant outlet pressure of "+str(pressure) )
    plt.xlabel('inlet liquid flow rate [m3/s]')
    plt.ylabel('inlet pressure [Pa]')
    plt.legend()
    plt.grid(True)
    fig.savefig('results/Inlet_pressures_mechanistic'+str(pressure)+'.png')

    #plot the error
    fig = plt.figure()
    plt.plot(ql_in_array, B_error, 'b',label="Bendiksen")
    plt.plot(ql_in_array, WG_error, 'r',label="W & G")
    plt.plot(ql_in_array, MLE_error, 'g', label="MLE")
    #plt.title("Error: Outlet pressure = "+str(pressure)+" Pa" )
    plt.xlabel('inlet liquid flow rate [m3/s]')
    plt.ylabel('Error [%]')
    plt.legend()
    plt.grid(True)
    fig.savefig('results/Error_mechanistic'+str(pressure)+'.png')
    
    print("Iterations complete")

test_model(dataset_p75, 750000)
test_model(dataset_p125, 1250000)


