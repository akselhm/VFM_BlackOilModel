import numpy as np
import matplotlib.pyplot as plt
from blackoil import black_oil
from Pipe import Pipe
from VFM import TH_model

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
N = 10000 

# thermal-hydraulic model
q_l_in = 0.157726   # [sm^3/s] 
p_out = 10e5    # [Pa] 
#void_frac_method = "Bendiksen" 
twophase_ff_method = "VG"


# -- initialize objects: fluid, pipe, TH_model --
fluid = black_oil(GOR, wc, p_bp, t_bp, rho_o0, rho_g0, rho_w0, gamma_g0, T_r)
pipe = Pipe(D, eps, length, N)
model = TH_model(fluid, pipe, q_l_in, p_out)

# run algorithm
B_pressures = model.Andreolli_algorithhm("Bendiksen", twophase_ff_method) 

WG_pressures = model.Andreolli_algorithhm("WG", twophase_ff_method)

# iterate through the pressures and find flow rates, wc, gor, wor etc. 
def computePipe(pressures, void_frac_method):
    WOR_list = np.zeros(len(pressures))
    wc_list = np.zeros(len(pressures))


    q_l_list = np.zeros(len(pressures))
    q_w_list = np.zeros(len(pressures))
    q_g_list = np.zeros(len(pressures))
    q_o_list = np.zeros(len(pressures))

    w_frac_list = np.zeros(len(pressures))
    void_frac_list = np.zeros(len(pressures))
    oil_frac_list = np.zeros(len(pressures))

    # standard conditions

    q_l0_list = np.zeros(len(pressures))
    q_w0_list = np.zeros(len(pressures))
    q_g0_list = np.zeros(len(pressures))
    q_o0_list = np.zeros(len(pressures))

    # iteration
    for i in range(len(pressures)):
        R_so = fluid.R_so_func(pressures[i])
        B_o = fluid.B_o_func(pressures[i], R_so)
        B_w = fluid.B_w_func(pressures[i])
        Z_g = fluid.Z_g_func(pressures[i])
        R_sl = fluid.R_so_func(R_so)
        B_l = fluid.B_l_func(B_w, B_o)
        
        rho_w = fluid.rho_w_func(B_w)
        rho_g = fluid.rho_g_func(pressures[i], Z_g) 
        rho_l = fluid.rho_l_func(R_sl, B_l) 
        rho_o = fluid.rho_o_func(R_so, B_o)
        
        visc_g = fluid.visc_g_func(rho_g) 
        visc_o = fluid.visc_o_func(pressures[i], R_so)
        visc_w = fluid.visc_w_func(pressures[i])
        
        B_g = fluid.rho_g0/rho_g     
        
        # find superficial velocities **
        j_g = B_g*model.j_o0*(fluid.GOR-R_so)    #TODO: change to local GOR (if local GOR should be used??)
        j_o = model.j_o0*B_o
        j_w = model.j_o0*fluid.get_local_WOR(B_w, B_o)*B_w
        j_l = j_o + j_w
        j   = j_l + j_g
        
        # void and liquid fractions
        if void_frac_method=="Bendiksen":
            Cd, Ud = model.Bendiksen(j)
        else: #woldesmayat and Gayar (prevously had bug for large N caused by P>P_pb.. =>.. R_so=GOR.. =>.. j_g=0)
            Cd, Ud = model.woldesemayat_ghajar(pressures[i], j, j_g, j_l, rho_g, rho_l, R_so) #woldesmayat and Gayar
        void_frac = model.void_frac_func(j, j_g, Cd, Ud) #*
        w_frac = model.liquid_k_frac(j_w, j_o, void_frac)
        o_frac = model.liquid_k_frac(j_o, j_w, void_frac)

        #save properties
        WOR_list[i] = fluid.get_local_WOR(B_w, B_o)
        wc_list[i] = fluid.get_local_wc(WOR_list[i])

        w_frac_list[i] = w_frac
        void_frac_list[i] = void_frac
        oil_frac_list[i] = o_frac

        q_l_list[i] = j_l*np.pi*pipe.D**2 / 4
        q_w_list[i] = j_w*np.pi*pipe.D**2 / 4
        q_g_list[i] = j_g*np.pi*pipe.D**2 / 4
        q_o_list[i] = j_o*np.pi*pipe.D**2 / 4

        # at SC

        q_g0_list[i] = q_g_list[i] / B_g + (q_o_list[i] * R_so) / B_o
        q_o0_list[i] = q_o_list[i]/B_o #+ (q_g_list[i] * R_so) / B_g # q_g_list[i]/(B_g*(fluid.GOR + 1/R_so))
        q_w0_list[i] = q_w_list[i]/B_w
        q_l0_list[i] = q_o0_list[i]+q_w0_list[i]

    
    return WOR_list, wc_list, w_frac_list, void_frac_list, oil_frac_list, q_l_list, q_w_list, q_g_list, q_o_list, q_l0_list, q_w0_list, q_g0_list, q_o0_list

BWOR_list, Bwc_list, Bw_frac_list, Bvoid_frac_list, Boil_frac_list, Bq_l_list, Bq_w_list, Bq_g_list, Bq_o_list, Bq_l0_list,  Bq_w0_list, Bq_g0_list, Bq_o0_list = computePipe(B_pressures, "Bendiksen")
WGWOR_list, WGwc_list, WGw_frac_list, WGvoid_frac_list, WGoil_frac_list, WGq_l_list, WGq_w_list, WGq_g_list, WGq_o_list, WGq_l0_list, WGq_w0_list, WGq_g0_list, WGq_o0_list = computePipe(B_pressures, "VG")

# print values at input and output for comparison
print("pressure at input with Bendiksen is :", B_pressures[0])
print("pressure at input with Woldesmayat and Gajar is :", WG_pressures[0])

print(" the pressure difference with Bendiksen is: ", (B_pressures[0]- B_pressures[-1])/100000, "bar")
print(" the pressure difference with W&G is: ", (WG_pressures[0]- WG_pressures[-1])/100000, "bar")

print("local oil flowrate at input and output is :", Bq_o_list[0], "and", Bq_o_list[-1])

print("local gas flowrate at input and output is :", Bq_g_list[0], "and", Bq_g_list[-1])

print("local water flowrate at input and output is :", Bq_w_list[0], "and", Bq_w_list[-1])

print("void fraction estimated at input and output with Bendiksen is :", Bvoid_frac_list[0], "and", Bvoid_frac_list[-1])
print("void fraction estimated at input and output with Woldesmayat and Gajar is :", WGvoid_frac_list[0], "and", WGvoid_frac_list[-1])

# ------- plotting ----------
x_list = pipe.x

# -- plot the pressure through the pipe --
fig = plt.figure()
plt.title('Pressure through pipe')
plt.plot(x_list, B_pressures, 'b--', label='Bendiksen')
plt.plot(x_list, WG_pressures, 'r-.', label='Woldesmayat and Ghajar')
plt.ylabel('Pressure [Pa]')
plt.xlabel('Position [m]')
plt.legend()
fig.savefig('results/pressureplot.png')

# -- void fraction --
fig = plt.figure()
plt.title('Void fraction through pipe')
plt.plot(x_list, Bvoid_frac_list, 'b--', label='Bendiksen')
plt.plot(x_list, WGvoid_frac_list, 'r-.', label='Woldesmayat and Ghajar')
plt.ylim(ymax = 1, ymin = 0)
plt.ylabel('Void fraction [-]')
plt.xlabel('Position [m]')
plt.legend()
fig.savefig('results/voidfractionplot.png')

fig = plt.figure()
plt.title('Void fraction through pipe (zoomed)')
plt.plot(x_list, Bvoid_frac_list, 'b--', label='Bendiksen')
plt.plot(x_list, WGvoid_frac_list, 'r-.', label='Woldesmayat and Ghajar')
#plt.ylim(ymax = 1, ymin = 0)
plt.ylabel('Void fraction [-]')
plt.xlabel('Position [m]')
plt.legend()
fig.savefig('results/voidfractionplotzoomed.png')

# -- oil fraction --
fig = plt.figure()
plt.title('Oil fraction through pipe')
plt.plot(x_list, Boil_frac_list, 'b--', label='Bendiksen')
plt.plot(x_list, WGoil_frac_list, 'r-.', label='Woldesmayat and Ghajar')
plt.ylim(ymax = 1, ymin = 0)
plt.ylabel('Oil Fraction [-]')
plt.xlabel('Position [m]')
plt.legend()
fig.savefig('results/oilfractionplot.png')

fig = plt.figure()
plt.title('Oil fraction through pipe (zoomed)')
plt.plot(x_list, Boil_frac_list, 'b--', label='Bendiksen')
plt.plot(x_list, WGoil_frac_list, 'r-.', label='Woldesmayat and Ghajar')
#plt.ylim(ymax = 1, ymin = 0)
plt.ylabel('Oil Fraction [-]')
plt.xlabel('Position [m]')
plt.legend()
fig.savefig('results/oilfractionplotzoomed.png')

# -- wc (should be constant) --
fig = plt.figure()
plt.title('Water Cut through pipe')
plt.plot(x_list, Bwc_list, 'b--', label='Bendiksen')
plt.plot(x_list, WGwc_list, 'r-.', label='Woldesmayat and Ghajar')
plt.ylim(ymax = 1, ymin = 0)
plt.ylabel('Water Cut[-]')
plt.xlabel('Position [m]')
plt.legend()
fig.savefig('results/wcplot.png')

fig = plt.figure()
plt.title('Water Cut through pipe (zoomed)')
plt.plot(x_list, Bwc_list, 'b--', label='Bendiksen')
plt.plot(x_list, WGwc_list, 'r-.', label='Woldesmayat and Ghajar')
plt.ylabel('Water Cut[-]')
plt.xlabel('Position [m]')
plt.legend()
fig.savefig('results/wcplotzoomed.png')

# -- total fractions through pipe --

fig = plt.figure()
plt.title('All phase fractions with best method')
plt.plot(x_list, WGw_frac_list, 'b', label='water fraction')
plt.plot(x_list, WGoil_frac_list, 'k', label='oil fraction')
plt.plot(x_list, WGvoid_frac_list, 'r', label='void fraction')
plt.ylim(ymax = 1, ymin = 0)
plt.ylabel('Fraction [-]')
plt.xlabel('Position [m]')
plt.legend()
fig.savefig('results/totalfractionplot.png')

# -- flowrates local conditions

fig = plt.figure()
plt.title('Water flowrate through pipe at local conditions')
plt.plot(x_list, Bq_w_list, 'b--', label='Bendiksen')
plt.plot(x_list, WGq_w_list, 'r-.', label='Woldesmayat and Ghajar')
plt.ylabel('Water flowrate [m^3/s]')
plt.xlabel('Position [m]')
plt.legend()
fig.savefig('results/waterflowrateplot.png')

# -- oil flowrate at SC --
fig = plt.figure()
plt.title('Oil flowrate through pipe at local conditions')
plt.plot(x_list, Bq_o_list, 'b--', label='Bendiksen')
plt.plot(x_list, WGq_o_list, 'r-.', label='Woldesmayat and Ghajar')
plt.ylabel('Oil flowrate [m^3/s]')
plt.xlabel('Position [m]')
plt.legend()
fig.savefig('results/oilflowrateplot.png')

# -- gas flowrate at SC --
fig = plt.figure()
plt.title('Gas flowrate through pipe at local conditions')
plt.plot(x_list, Bq_g_list, 'b--', label='Bendiksen')
plt.plot(x_list, WGq_g_list, 'r-.', label='Woldesmayat and Ghajar')
plt.ylabel('Oil flowrate [m^3/s]')
plt.xlabel('Position [m]')
plt.legend()
fig.savefig('results/gasflowrateplot.png')


# --- flowrates standard conditions ---

# -- water flowrate at SC --
fig = plt.figure()
plt.title('Water flowrate through pipe at SC')
plt.plot(x_list, Bq_w0_list, 'b--', label='Bendiksen')
plt.plot(x_list, WGq_w0_list, 'r-.', label='Woldesmayat and Ghajar')
plt.ylabel('Water flowrate [sm^3/s]')
plt.xlabel('Position [m]')
plt.legend()
fig.savefig('results/waterflowrateSCplot.png')

# -- oil flowrate at SC --
fig = plt.figure()
plt.title('Oil flowrate through pipe at SC')
plt.plot(x_list, Bq_o0_list, 'b--', label='Bendiksen')
plt.plot(x_list, WGq_o0_list, 'r-.', label='Woldesmayat and Ghajar')
plt.ylabel('Oil flowrate [sm^3/s]')
plt.xlabel('Position [m]')
plt.legend()
fig.savefig('results/oilflowrateSCplot.png')

# -- gas flowrate at SC --
fig = plt.figure()
plt.title('Gas flowrate through pipe at SC')
plt.plot(x_list, Bq_g0_list, 'b--', label='Bendiksen')
plt.plot(x_list, WGq_g0_list, 'r-.', label='Woldesmayat and Ghajar')
plt.ylabel('Oil flowrate [sm^3/s]')
plt.xlabel('Position [m]')
plt.legend()
fig.savefig('results/gasflowrateSCplot.png')

