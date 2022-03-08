#file for loading and pre-processing the OLGA data
#only the ppl-files are used
#tested for one case only, will be expanded to multiple cases

# variables that should be used for 

import pyfas as fa
import pandas as pd
import subprocess #to run batchfile
import matplotlib.pyplot as plt
import numpy as np
import os
pd.options.display.max_colwidth = 120

# - load file - 

ppl_path = 'C:/Users/aksel/NTNU/Prosjektoppgave/OLGA-filer/' # fullstendig bane: C:\Users\aksel\NTNU\Prosjektoppgave\OLGA-filer
inpfilename = 'OLGAinputPython.inp'
pplfilename = 'inputOLGAtest.ppl'
ppl = fa.Ppl(ppl_path+pplfilename) 

#TODO: generate data: 1) modify input.txt file and rename to input.inp 2) run modified input file via bat file 3) extract results and add to dataset

# STEP 1) modify input file, copy, read as string

inputfile = open(ppl_path+pplfilename, "r")
lines = inputfile.readlines()
lines[46] = "changed line"


inputfile = open(ppl_path+pplfilename, "w")
inputfile.writelines(lines)
#print()
print(lines[3:5])
print(lines[45:47])

inputfile.close()
exit()
# STEP 2) run modified input file from bat-file

#subprocess.call([r'C:/Users/aksel/NTNU/Prosjektoppgave/OLGA-filer/testbatchfile.bat'])
subprocess.call([r'C:/Users/aksel/NTNU/Prosjektoppgave/VFM_BlackOilModel/testbatchfile.bat'])
# current error: tries to run the inp-file from "this" folder and not the folder defined in the bat-file
#possible solution: relocate bat-file and input file to current folder. (also modify the "call" to reflect this folder) SUCSESS



# - filter trends - 

#print(ppl.filter_data("PT"))

#print(pd.DataFrame(ppl.filter_data('PT'), index=("Profiles",)).T)

# - Extract data -

ppl.extract(15) #extract the pressures to the ppl object
pressures = ppl.data[15][1] # save the pressures to array

print(pressures[-1]) #print the pressures from the last time step


# -- Sceleton-code to the full task of creating the dataset --

size = 5            #size of dataset (Expand when it works)

dataset = np.zeros((size, 3)) #3 is here number of variables in the dataset. Change when you know this

#define range of the variables
min_qo_in = 0.01    #change when you know what values you want
max_qo_in = 0.1     #change when you know what values you want
min_p_out = 10e4    # --""--
max_p_out = 50e5    # --""--

for i in range(size):
    qo_in = np.random.uniform(min_qo_in, max_qo_in)
    p_out = np.random.uniform(min_qo_in, max_qo_in)
    #DO STEP ONE: modify input file to have new flowrate and pressure (and other parameters if wanted)
    #TODO: ..

    #STEP TWO: run the modified input file
    subprocess.call([r'C:/Users/aksel/NTNU/Prosjektoppgave/VFM_BlackOilModel/testbatchfile.bat']) #run inputfile via batchfile

    #STEP THREE: extract data and add to dataset
    ppl = fa.Ppl(ppl_path+pplfilename) #create ppl-object (change file name if necessary)
    
    ppl.extract(15) #extract the pressures to the ppl object
    pressures = ppl.data[15][1] # save the pressures to array
    p_in = pressures[-1][1] #last timestep "first" value (disregard value at pos. 0)

    #TODO: extract all other values of interest

    #make array of values
    arr = np.array([p_in, p_out, qo_in]) #TODO: add additional values

    #add values to dataset
    dataset[i] = arr
    