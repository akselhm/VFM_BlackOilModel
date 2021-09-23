import numpy as np
from blackoil import blackoil



# --- Variables ---

P_atm = 100 #double check 


# input data from tab 2 and 3

API = 29.05         #API gravity (denoted gamma_API in tab)
gamma_g0 = 0.83     #gas specific gravity at SC
T_r = 333           #reservoir temperature
P_b = 340*10**5     # bubble pressure at T_r
GOR =196.4          #gas-oil ratio
GORI =[]            #injection gas oil-ratio
WOR =15.6           #water-oil ratio
P_plat =22.2*10**5  # pressure at platform
P_WCT =104.6*10**5  # pressure at WCT (used for testing)
T_k =T_r            # temperature at WCT 
j_o0 = 0.6860       # oil superficial velocity at SC
eps = 0.00021       # roughness
D = 0.1524          # pipe diameter


# ---

gamma_g = gamma_g0 # from eq. 30

theta = 0 #angle relative to the horizontal plane

N = 10000
#x = np.zeros(22) #insert from tab 1
z = np.linspace(0, 1000, N) 

delta=10**(-4)
maxit = 100
#define under-relaxation factor
urf = 0.1

