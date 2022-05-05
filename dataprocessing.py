#file for creating, loading and pre-processing the OLGA data
#uses an inp-file to create an OLGA-model and a bat-file to run the case. Retrieves the data from the generated ppl-file. The inp-file is modified between each run in a loop.

#OPTIONS
change_pressure=False #deside if input pressure should be changed
skewed_ql_in = False #decide if data set should be evenly distibuted w.r.t inlet liquid (oil) flowrate or not. Example of skewed dataset: 10% - [0.05 - 0.15] m3/s and 90% - [0.15 - 0.25] m3/s
add_noise = False

#TODO: select what parameters is wanted from the ppl-file and expand size of dataset
#TODO: add noise to dataset (!!!)


import pyfas as fa  # library to extract results from ppl file
import pandas as pd
import subprocess   # to run batchfile
import matplotlib.pyplot as plt #not needed atm
import numpy as np
pd.options.display.max_colwidth = 120 #needed(?) for the dataframes created from ppl-objects (pyfas)

# - load files - 

path = 'C:/Users/aksel/NTNU/Prosjektoppgave/VFM_BlackOilModel/' # .. do not really need anymore as all files are in same folder 
inpfilename = 'inputOLGAtest.inp' # name of input file used to create OLGA-object
pplfilename = 'inputOLGAtest.ppl'   #name of output file


# Steps for generating data: 1) modify input file with new variables 2) run modified input file via bat file 3) extract results from ppl file and add to dataset

size = 20            #size of dataset
number_of_vars = 10  #number of variables to extract to the dataset. Change when you know this
column_names = ["inlet pressure [Pa]", 
    "outlet pressure [Pa]", 
    "inlet oil flowrate [m3/s]", 
    "outlet oil flowrate [m3/s]",
    "inlet gas flowrate [m3/s]", 
    "outlet gas flowrate [m3/s]",
    "inlet water flowrate [m3/s]", 
    "outlet water flowrate [m3/s]",
    "inlet holdup [-]",
    "outlet holdup [-]"] #change if changing the variables selected


dataset = np.zeros((size, number_of_vars)) 



#define range of the variables that chould be changed
wc = 0.3 #water cut (use same as in OLGA-file)
min_ql_in = 0.05    # [Sm3/s] 
max_ql_in = 0.25     # [Sm3/s]
if change_pressure == True:
    min_p_out = 5*10**5     # [Pa] (deside on range)
    max_p_out = 15*10**5     # [Pa] 

#make skewed data set so that: x% - [0.05 - 0.15] m3/s and (100-x)% - [0.15 - 0.25] m3/s
if skewed_ql_in== True:
    split = 0.1 #determines how large portion of the data set should be in lower range of the values
    max_ql_in_low = 0.15    # [Sm3/s] maximum value for low input flowrate dataset
    min_ql_in_high = 0.15   # [Sm3/s] minmum value for high input flowrate dataset

# -- Loop for generating data --
for i in range(size):

    print("Iteration at: ", i, " out of ", size)
    #Generate random inlet oil flowrate and outlet pressure

    #Flowrate
    if skewed_ql_in == True:    #make skewed dataset
        distribution_parameter = np.random.uniform(0, 1)
        if distribution_parameter < split: #generate data in the lower range of ql_in
            ql_in = np.random.uniform(min_ql_in, max_ql_in_low)
        else:   #generate data in the hogher range of ql_in
            ql_in = np.random.uniform(min_ql_in_high, max_ql_in)
    else:   #make evenly distributed values for ql_in
        ql_in = np.random.uniform(min_ql_in, max_ql_in)
    #Change from liquid inlet flow rate to oil inlet flow rate
    qo_in0 = (1-wc)*ql_in #standard inlet oil flow rate

    #Pressure
    if change_pressure ==True:
        p_out = np.random.uniform(min_p_out, max_p_out)

    # -- STEP ONE: modify input file to have new flowrate and pressure (and other parameters if wanted) --
    
    inputfile = open(path+inpfilename, "r")
    lines = inputfile.readlines()
    #change inlet oil flowrate
    lines[47] = "        FEEDNAME=Bofluid, PHASE=OIL, FEEDSTDFLOW={} Sm3/s \n".format(qo_in0)
    #change outlet pressure
    if change_pressure == True:
        lines[70] = "        TEMPERATURE=25 C, PRESSURE={} Pa, FLUID=Bofluid \n".format(p_out) 
    inputfile.close()

    inputfile = open(path+inpfilename, "w")
    inputfile.writelines(lines)
    inputfile.close()

    # -- STEP TWO: run the modified input file from batch file --
    subprocess.call([r'C:/Users/aksel/NTNU/Prosjektoppgave/VFM_BlackOilModel/testbatchfile.bat']) #run inputfile via batchfile

    # -- STEP THREE: extract data and add to dataset (keep control of how many variables you have in each ) --
    ppl = fa.Ppl(path+pplfilename) #create ppl-object 
    
    # pressure
    ppl.extract(15) #extract the pressures to the ppl object
    pressures = ppl.data[15][1] # save the pressures to array
    p_in = pressures[-1][0] #last timestep
    #if change_pressure == False: #<-remove this? 
    p_out = pressures[-1][-1] # or give the specified value directly if known in advanced (as it is not supposed to change) (NOTE: outlet in ppl is not giving the same value as specified in the inp-file)

    # Flow rates (extract at local conditions, not standard)
    # gas voulme flow
    ppl.extract(3)
    gas_flow = ppl.data[3][1] # save the pressures to array
    qg_in = gas_flow[-1][1] #last timestep, "first" value (disregard value at pos. 0 for flow rates)
    qg_out = gas_flow[-1][-1] #last timestep, last value 

    # Volumetric flow rate oil
    ppl.extract(5)
    oil_flowrate = ppl.data[5][1] # save the pressures to array
    qo_in = oil_flowrate[-1][1] #last timestep, first value 
    qo_out = oil_flowrate[-1][-1] #last timestep, last value 

    # Volumetric flow rate water
    ppl.extract(6)
    water_flowrate = ppl.data[6][1] # save the pressures to array
    qw_in = water_flowrate[-1][1] #last timestep, last value
    qw_out = water_flowrate[-1][-1] #last timestep, last value 

    # Holdup (use to derive void fraction for MLE)
    ppl.extract(13)
    holdup = ppl.data[13][1]    # save the holdup to array
    hol_in = holdup[-1][0]      # last timestep, first value
    hol_out = holdup[-1][-1]    # last timestep, last value

    #make array of values
    arr = np.array([p_in, p_out, qo_in, qo_out, qg_in, qg_out, qw_in, qw_out, hol_in, hol_out]) #TODO: add additional values if needed

    #add values to dataset
    dataset[i] = arr

if add_noise: #noise should be added for the MLE to work !!
    # - determin maximum error for each entity (99.73 percent of all values is within this, meaning it is 3*std_dev)
    maximum_noise = 0.05 #percent of maximum value
    #print(dataset)
    max_p_error = maximum_noise * np.max(dataset[:,0]) # 5 percent of maximum of the measured inlet pressures
    max_qo_error = maximum_noise * np.max(dataset[:,2]) # 5 percent of maximum of the measured inlet oil flow rate
    max_qg_error = maximum_noise * np.max(dataset[:,5]) # 5 percent of maximum of the measured outlet gas flow rate
    max_qw_error = maximum_noise * np.max(dataset[:,6]) # 5 percent of maximum of the measured inlet water flow rate
    max_hu_error = maximum_noise * np.max(dataset[:,8]) # 5 percent of maximum of the measured inlet holdup
    # - make error arrays and ensure 99.73 percent of all values is within this maximum error. Assume homoscedasticity
    # inlet
    noise_p_in = np.random.normal(0, max_p_error/3, size) 
    noise_qo_in = np.random.normal(0,max_qo_error/3, size)
    noise_qg_in = np.random.normal(0,max_qg_error/3, size)
    noise_qw_in = np.random.normal(0,max_qw_error/3, size)
    noise_hu_in = np.random.normal(0,max_hu_error/3, size)
    # outlet
    noise_p_out = np.random.normal(0, max_p_error/3, size) 
    noise_qo_out = np.random.normal(0,max_qo_error/3, size)
    noise_qg_out = np.random.normal(0,max_qg_error/3, size)
    noise_qw_out = np.random.normal(0,max_qw_error/3, size)
    noise_hu_out = np.random.normal(0,max_hu_error/3, size)
    # - add the noise/error to the dataset array
    dataset[:,0] += noise_p_in
    dataset[:,1] += noise_p_out
    dataset[:,2] += noise_qo_in
    dataset[:,3] += noise_qo_out
    dataset[:,4] += noise_qg_in 
    dataset[:,5] += noise_qg_out
    dataset[:,6] += noise_qw_in
    dataset[:,7] += noise_qw_out
    dataset[:,8] += noise_hu_in
    dataset[:,9] += noise_hu_out

# create dataframe from numpy array with dataset
df = pd.DataFrame(data = dataset, columns = column_names)

# Generate a csv file from the dataframe and save to data-folder
df.to_csv("data/dataset.csv", index=False)
