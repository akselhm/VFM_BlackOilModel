import numpy as np
import matplotlib.pyplot as plt
from blackoil import black_oil
from Pipe import Pipe
from VFM import TH_model

# -- initialize values --

# black oil
GOR = 50.0  #[-]
wc = 0.30   #[-]
p_bp = 50.0e5   #[Pa] ?
t_bp = 20.0 + 273.15 # [K]
rho_o0 = 1100 #867.0  #[..] unit 
rho_g0 = 0.997
rho_w0 = 1200.0
gamma_g0 = 0.799
T_r = 25 + 273.15    # [K] included in black oil as it is constant

#pipe
D = 0.2     # [m]
eps = 5e-3# 0.005 # [m]
length = 5000   # [m]
N = 100000 

# thermal-hydraulic model
q_l_in = 1.5*0.157726   # [..] unit
p_out = 10e5    # [Pa] ?
#void_frac_method = "Bendiksen" 
twophase_ff_method = "VG"

# -- initialize objects: fluid, pipe, TH_model --
fluid = black_oil(GOR, wc, p_bp, t_bp, rho_o0, rho_g0, rho_w0, gamma_g0, T_r)
pipe = Pipe(D, eps, length, N)
model = TH_model(fluid, pipe, q_l_in, p_out)

#run algorithm
B_pressures = model.Andreolli_algorithhm("Bendiksen", twophase_ff_method) 
#B_pressures = B_pressures[1:] # an error so that the last step is not changed TODO: fix error and remove this

WG_pressures = model.Andreolli_algorithhm("WG", twophase_ff_method)
#WG_pressures = WG_pressures[1:]

#TODO: -- calculate all other values from the pressures --
#iterate through the pressures and find flow rates, wc, gor, wor etc. 
def computePipe(pressures, void_frac_method):
    WOR_list = np.zeros(len(pressures))
    wc_list = np.zeros(len(pressures))


    q_l_list = np.zeros(len(pressures))
    q_w_list = np.zeros(len(pressures))
    q_g_list = np.zeros(len(pressures))
    q_o_list = np.zeros(len(pressures))

    void_frac_list = np.zeros(len(pressures))
    oil_frac_list = np.zeros(len(pressures))

    for i in range(len(pressures)):
        R_so = fluid.R_so_func(pressures[i])
        B_o = fluid.B_o_func(pressures[i], R_so)
        B_w = fluid.B_w_func(pressures[i])
        Z_g = fluid.Z_g_func(pressures[i])
        R_sl = fluid.R_so_func(R_so)
        B_l = fluid.B_l_func(B_w, B_o)
        
        rho_w = fluid.rho_w_func(B_w)
        rho_g = fluid.rho_g_func(pressures[i], Z_g) #*
        rho_l = fluid.rho_l_func(R_sl, B_l) #*
        rho_o = fluid.rho_o_func(R_so, B_o)
        
        visc_g = fluid.visc_g_func(rho_g) #**
        visc_o = fluid.visc_o_func(pressures[i], R_so)
        visc_w = fluid.visc_w_func(pressures[i])
        
        B_g = fluid.rho_g0/rho_g      # gas formation volume factor (function?)
        
        # find superficial velocities **
        j_g = B_g*model.j_o0*(fluid.GOR-R_so)    #TODO: change to local GOR (if local GOR should be used??)
        j_o = model.j_o0*B_o
        j_w = model.j_o0*fluid.get_local_WOR(B_w, B_o)*B_w
        j_l = j_o + j_w
        j   = j_l + j_g
        
        # void and liquid fractions
        if void_frac_method=="Bendiksen":
            Cd, Ud = model.Bendiksen(j)
        else: #woldesmayat and Gayar (currently bug for large N caused by P>P_pb.. =>.. R_so=GOR.. =>.. j_g=0)
            Cd, Ud = model.woldesemayat_ghajar(pressures[i], j, j_g, j_l, rho_g, rho_l, R_so) #woldesmayat and Gayar
        void_frac = model.void_frac_func(j, j_g, Cd, Ud) #*
        w_frac = model.liquid_k_frac(j_w, j_o, void_frac)
        o_frac = model.liquid_k_frac(j_o, j_w, void_frac)

        #save properties
        WOR_list[i] = fluid.get_local_WOR(B_w, B_o)
        wc_list[i] = fluid.get_local_wc(WOR_list[i])

        void_frac_list[i] = void_frac
        oil_frac_list[i] = o_frac

        q_l_list[i] = j_l*np.pi*pipe.D**2 / 4
        q_w_list[i] = j_w*np.pi*pipe.D**2 / 4
        q_g_list[i] = j_g*np.pi*pipe.D**2 / 4
        q_o_list[i] = j_o*np.pi*pipe.D**2 / 4
    
    return WOR_list, wc_list, void_frac_list, oil_frac_list, q_l_list, q_w_list, q_g_list, q_o_list

BWOR_list, Bwc_list, Bvoid_frac_list, Boil_frac_list, Bq_l_list, Bq_w_list, Bq_g_list, Bq_o_list = computePipe(B_pressures, "Bendiksen")
WGWOR_list, WGwc_list, WGvoid_frac_list, WGoil_frac_list, WGq_l_list, WGq_w_list, WGq_g_list, WGq_o_list = computePipe(B_pressures, "VG")

# ------- plotting ----------
x_list = pipe.x
# -- plot the pressure through the pipe --
fig = plt.figure()
plt.plot(x_list, B_pressures, 'b--', label='Bendiksen')
plt.plot(x_list, WG_pressures, 'r-.', label='Woldesmayat and Ghajar')
plt.ylabel('Pressure [Pa]')
plt.xlabel('Position [m]')
plt.legend()
fig.savefig('results/pressureplot.png')

# -- void fraction --
fig = plt.figure()
plt.plot(x_list, Bvoid_frac_list, 'b--', label='Bendiksen')
plt.plot(x_list, WGvoid_frac_list, 'r-.', label='Woldesmayat and Ghajar')
plt.ylim(ymax = 1, ymin = 0)
plt.ylabel('Void fraction [Pa]')
plt.xlabel('Position [m]')
plt.legend()
fig.savefig('results/voidfractionplot.png')

# -- oil fraction --
fig = plt.figure()
plt.plot(x_list, Boil_frac_list, 'b--', label='Bendiksen')
plt.plot(x_list, WGoil_frac_list, 'r-.', label='Woldesmayat and Ghajar')
plt.ylim(ymax = 1, ymin = 0)
plt.ylabel('Oil Fraction [-]')
plt.xlabel('Position [m]')
plt.legend()
fig.savefig('results/oilfractionplot.png')

# -- wc (should be constant) --
fig = plt.figure()
plt.plot(x_list, Bwc_list, 'b--', label='Bendiksen')
plt.plot(x_list, WGwc_list, 'r-.', label='Woldesmayat and Ghajar')
plt.ylim(ymax = 1, ymin = 0)
plt.ylabel('Water Cut[-]')
plt.xlabel('Position [m]')
plt.legend()
fig.savefig('results/wcplot.png')