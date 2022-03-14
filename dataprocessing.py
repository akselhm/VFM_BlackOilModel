#file for creating, loading and pre-processing the OLGA data
#uses an inp-file to create an OLGA-model and a bat-file to run the case. Retrieves the data from the generated ppl-file. The inp-file is modified between each run in a loop.

import pyfas as fa
import pandas as pd
import subprocess #to run batchfile
import matplotlib.pyplot as plt
import numpy as np
import os
pd.options.display.max_colwidth = 120

# - load files - 

ppl_path = 'C:/Users/aksel/NTNU/Prosjektoppgave/OLGA-filer/' # fullstendig bane: C:\Users\aksel\NTNU\Prosjektoppgave\OLGA-filer
path = 'C:/Users/aksel/NTNU/Prosjektoppgave/VFM_BlackOilModel/' #do not need anymore as all files are in same folder
inpfilename = 'inputOLGAtest.inp'#'OLGAinputPython.txt' #OLGA inp.-file as .txt-file (change name if needed)
pplfilename = 'inputOLGAtest.ppl'
#ppl = fa.Ppl(ppl_path+pplfilename) 

#TODO: generate data: 1) modify input.txt file and rename to input.inp 2) run modified input file via bat file 3) extract results and add to dataset

# -- Sceleton-code to the full task of creating the dataset --

size = 5            #size of dataset (Expand when it works)
number_of_vars = 6  #number of variables to extract to the dataset. Change when you know this
column_names = ["inlet pressure [bara]", "outlet pressure [bara]", "oil inlet flowrate [Sm3/s]", "oil outlet flowrate [Sm3/s]", "gas outlet flowrate [Sm3/s]", "water outlet flowrate [Sm3/s]"] #change if changing the variables selected


dataset = np.zeros((size, number_of_vars)) 

#define range of the variables
min_qo_in = 0.01    # [sm3/s] change when you know what values you want
max_qo_in = 0.1     # [sm3/s] change when you know what values you want
min_p_out = 10      # [bara] --""-- (change to pascal)
max_p_out = 50      # [bara] --""--

# -- Loop for generating data --
for i in range(size):
    #generate random inlet oil flowrate and outlet pressure
    qo_in = np.random.uniform(min_qo_in, max_qo_in)
    p_out = np.random.uniform(min_p_out, max_p_out)

    # -- STEP ONE: modify input file to have new flowrate and pressure (and other parameters if wanted)
    
    inputfile = open(path+inpfilename, "r")
    lines = inputfile.readlines()
    #change inlet oil flowrate
    lines[47] = "        FEEDNAME=Bofluid, PHASE=OIL, FEEDSTDFLOW={} Sm3/s \n".format(qo_in)
    #change outlet pressure
    lines[70] = "        TEMPERATURE=25 C, PRESSURE={} bara, FLUID=Bofluid \n".format(p_out) 
    inputfile.close()

    inputfile = open(path+inpfilename, "w")
    inputfile.writelines(lines)
    inputfile.close()

    # -- STEP TWO: run the modified input file from batch file
    subprocess.call([r'C:/Users/aksel/NTNU/Prosjektoppgave/VFM_BlackOilModel/testbatchfile.bat']) #run inputfile via batchfile

    # -- STEP THREE: extract data and add to dataset (keep control of how many variables you have in each )
    ppl = fa.Ppl(path+pplfilename) #create ppl-object (change file name if necessary)
    
    # pressure
    ppl.extract(15) #extract the pressures to the ppl object
    pressures = ppl.data[15][1] # save the pressures to array
    p_in = pressures[-1][1] #last timestep "first" value (disregard value at pos. 0)

    # gas voulme flow
    ppl.extract(3)
    gas_flow = ppl.data[3][1] # save the pressures to array
    qg_out = gas_flow[-1][-1] #last timestep last value 

    # Volumetric flow rate oil
    ppl.extract(5)
    oil_flowrate = ppl.data[5][1] # save the pressures to array
    qo_out = oil_flowrate[-1][-1] #last timestep last value 

    # Volumetric flow rate water
    ppl.extract(5)
    water_flowrate = ppl.data[5][1] # save the pressures to array
    qw_out = water_flowrate[-1][-1] #last timestep last value 

    #make array of values

    arr = np.array([p_in, p_out, qo_in, qo_out, qg_out, qw_out]) #TODO: add additional values if needed

    #add values to dataset
    dataset[i] = arr

print(dataset)

# create dataframe from numpy array with dataset
df = pd.DataFrame(data = dataset,
columns = column_names)

# Generate a csv file from the dataframe
df.to_csv("data/dataset.csv")


exit()

# -- Testing all steps --
# STEP 1) modify input file, copy, read as string

inputfile = open(ppl_path+inpfilename, "r")
lines = inputfile.readlines()
lines[46] = "        FEEDNAME=Bofluid, PHASE=OIL, FEEDSTDFLOW={} Sm3/s \n".format(0.5) #change 0.5 to the flowrate that is desired
inputfile.close()

inputfile = open(ppl_path+inpfilename, "w")
inputfile.writelines(lines)
inputfile.close()

"""
inputfile = open(path+pplfilename, "r") #change path later ...
lines = inputfile.readlines()
lines[46] = "changed line" #test if the changes are made
#lines[46]= "PARAMETERS LABEL=NODE_3, TYPE=MASSFLOW, FEEDNAME=Bofluid, PHASE=OIL, FEEDSTDFLOW={} Sm3/s, \ \n".format(feedflow_std[i])
inputfile.close()

inputfile = open(ppl_path+pplfilename, "w")
inputfile.writelines(lines)
#print()
print(lines[3:5])
print(lines[45:47])

inputfile.close()

#rename files from .txt to .inp (adapted from Md Rizwan) NOT NEEDED ANYMORE
folder = "C:/Users/aksel/NTNU/Prosjektoppgave/VFM_BlackOilModel"
filename = 'OLGAinputPython.txt'
for filename in os.listdir(folder):
    infilename = os.path.join(folder,filename)
    if not os.path.isfile(infilename): continue
    oldbase = os.path.splitext(filename)
    newname = infilename.replace('.txt', '.inp')
    output = os.rename(infilename, newname)
"""

# STEP 2) run modified input file from bat-file

#subprocess.call([r'C:/Users/aksel/NTNU/Prosjektoppgave/OLGA-filer/testbatchfile.bat'])
subprocess.call([r'C:/Users/aksel/NTNU/Prosjektoppgave/VFM_BlackOilModel/testbatchfile.bat'])
# current error: tries to run the inp-file from "this" folder and not the folder defined in the bat-file
#possible solution: relocate bat-file and input file to current folder. (also modify the "call" to reflect this folder) SUCSESS


exit()
# - filter trends - 

#print(ppl.filter_data("PT"))

#print(pd.DataFrame(ppl.filter_data('PT'), index=("Profiles",)).T)

# - Extract data -

ppl.extract(15) #extract the pressures to the ppl object
pressures = ppl.data[15][1] # save the pressures to array

print(pressures[-1]) #print the pressures from the last time step


    