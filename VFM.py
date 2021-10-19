import numpy as np
import matplotlib.pyplot as plt
from blackoil import my_fluid



# --- Variables ---

# input data from tab 2 and 3 (do not need here??)

GOR =196.4          #gas-oil ratio
WOR =15.6           #water-oil ratio
P_plat =22.2*10**5  # pressure at platform
P_WCT =104.6*10**5  # pressure at WCT (used for testing)
j_o0 = 0.6860       # oil superficial velocity at SC
eps = 0.00021       # roughness
D = 0.1524          # pipe diameter

g_s = 9.81      #gravity acceleration
# ---

theta = 0 #angle relative to the horizontal plane

N = 10000
#x = np.zeros(22) #insert from tab 1
z = np.linspace(0, 1000, N) 

delta=10**(-4)
maxit = 100
#define under-relaxation factor
urf = 0.1



def massFlux_func(rho_l, rho_g, j_l, j_g):
    #denoted G in the paper (eq 9)
    return rho_l*j_l + rho_g*j_g 


def void_frac_func(j, j_g, Cd, Ud):
    #based on the drift-flux model (Zuber and Findlay, 1965) (eq 11)
    return j_g/(Cd*j+Ud)

def liquid_k_frac(j_k, j_sub, void_frac):
    # function to calculate fraction of oil or water (eq 38 and 39)
    # Ex: if calculating oil fraction, input should be (j_o, j_w,...)
    return j_k/(j_k+j_sub)*(1-void_frac)


#discretization

def pressureStep(i, P_i, rho_m_avg, Z_avg):
    # function for finding the pressure uptream P_(i-1) (eq 40)
    # here we set Z = phi*f_l*G**2/(rho_l*D) for simplicity
    return P_i + rho_m_avg*g_s*(z[i]-z[i-1]) + 0.5*Z_avg


# --- equations from Appendix B (two-phase flow multiplier correlations) ---

def DarcyFrictionFactor(Re):
    # Darcy friction factor f
    f = ((64/Re)**8 + 9.5*(np.log(eps/(3.7*D) + 5.74/(Re**0.9)) - (2500/Re)**6)**(-16))**0.125
    return f

def VieraGarcia(f, f_l, rho, rho_l):
    #return: two-phase multiplier
    # what friction factor f and density rho is inserted depends on what approach is used
    return (f/f_l)*(rho_l/rho)



def MullerHeck(f_l, f_g, rho_l, rho_g, G, x):
    #return: two-phase multiplier
    A = 0.5*f_l*G**2/(rho_l*D)
    B = 0.5*f_g*G**2/(rho_g*D)
    Gc = A+ 2*(B-A)*x
    return (Gc*(1-x)**(1/3) + B*x**3)/A

# --- equations from Appendix C (Void fraction correlations) ---


def Cd_and_Ud_func(P, j, j_g, j_l, rho_g, rho_l, angle, method="B"): #might split to three funcs
    # Drift flux correlations
    # return: Cd, Ud
    # ---------------------------------------
    Cd, Ud = 0, 0
    # default method Bendiksen(1984)
    Fr_j = j/(np.sqrt(g_s*D))
    if Fr_j < 3.5:
        Cd = 1.05 + 0.15*np.sin(angle)
        Ud = np.sqrt(g_s*D)*(0.35*np.sin(angle) + 0.54*np.cos(angle))
    else: #Fr_j >= 3.5
        Cd = 12
        Ud = 0.35*np.sqrt(g_s*D)*np.sin(angle)
    return Cd, Ud


def run(blackoilmodel):
    
    P_list = np.zeros(N) # list for saving the pressure at each step
    
    #begin with the initial values at the platform
    P_i = P_plat
    
    #initialize some values
    rho_m_i = 0 
    Z_i = 0
    for i in range(N-1,-1,-1):  #find upstream pressure (iterate backwards)
        #--define initial guess P_(i-1)^pred=P_i
        P_pred=P_i
        epsilon = 1000  #initialize epsilon to start iteration
        it=0 # make sure the loop is not eternal
        while epsilon>delta:
        
            # black-oil correlations, viscosities, and densities (make one function in myfluid-class to compute all)
            R_so = blackoilmodel.R_so_func(P_pred)
            B_o = blackoilmodel.B_o_func(P_pred, R_so)
            B_w = blackoilmodel.B_w_func(P_pred)
            Z_g = blackoilmodel.Z_g_func(P_pred)
            R_sl = blackoilmodel.R_so_func(R_so)
            B_l = blackoilmodel.B_l_func(B_w, B_o)
            
            rho_w = blackoilmodel.rho_w0/B_w    #function?
            rho_g = blackoilmodel.rho_g_func(P_pred, Z_g)
            rho_l = blackoilmodel.rho_l_func(R_sl, B_l)
            rho_o = blackoilmodel.rho_o_func(R_so, B_o)
            
            visc_g = blackoilmodel.visc_g_func(rho_g)
            visc_o = blackoilmodel.visc_o_func(P_pred, R_so)
            visc_w = blackoilmodel.visc_w_func(P_pred)
            
            B_g = blackoilmodel.rho_g0/rho_g      # gas formation volume factor (function?)
            
            # find superficial velocities
            j_g = B_g*j_o0*(GOR-R_so)
            j_o = j_o0*B_o
            j_w = j_o0*WOR*B_w
            j_l = j_o + j_w
            j   = j_l + j_g
            
            # void and liquid fractions
            Cd, Ud = Cd_and_Ud_func(P_pred, j, j_g, j_l, rho_g, rho_l, theta, method='B') #Bendiksen (have not implemented the others)
            void_frac = void_frac_func(j, j_g, Cd, Ud)
            w_frac = liquid_k_frac(j_w, j_o, void_frac)
            o_frac = liquid_k_frac(j_o, j_w, void_frac)
            
            # additional mixed properties (density and viscosity)
            rho_m = rho_g*void_frac + rho_o*o_frac + rho_w*w_frac
            visc_l = (visc_o*o_frac + visc_w*w_frac)/(o_frac + w_frac) #double check
            visc_m = visc_g*void_frac + visc_o*o_frac + visc_w*w_frac
            
            G = rho_l*j_l + rho_g*j_g   #total mass flux
            
            # find friction factors        
            Re_l = G*D/visc_l
            f_l = DarcyFrictionFactor(Re_l)
            
            # find two phase multiplier (Vierra and Garcia or Muller and Heck)
            # "VGm"(default), center of mass
            v_m = (rho_g*j_g + rho_o*j_o + rho_w*j_w)/(rho_g*void_frac + rho_g*void_frac + rho_g*void_frac) #double check this eq
            rho = rho_m
            Re_cm = rho_m*v_m*D/visc_m
            f = DarcyFrictionFactor(Re_cm)
            phi = VieraGarcia(f, f_l, rho, rho_l)
                    
            
            # compute averages
            if i<N-1:
                rho_m_avg = 0.5*(rho_m+rho_m_i) #ajust to 0.5*(rho_m+rho_m_pred) (not sure how to do it for the first step)
            
                Z=phi*f_l*G**2/(rho_l*D)
                Z_avg = 0.5*(Z + Z_i) 
                
            else:   # initial step
                rho_m_avg = rho_m #ajust to 0.5*(rho_m+rho_m_pred) (not sure how to do it for the first step)
                
                Z=phi*f_l*G**2/(rho_l*D)
                Z_avg = Z #ajust 
                
            
            #find pressure
            P = pressureStep(i, P_i, rho_m_avg, Z_avg)
             
            # P iteration
            # check convergence (compute epsilon)
            epsilon = abs(P_pred-P)/P
            
            # under relaxed predicted value
            #check covergence (epsilon)
            P_pred += urf*(P-P_pred)
            
            #make sure the loop is not eternal
            it+=it
            if it>maxit:
                print("max iterations reached. not convergent for P at step", i)
                return
        
        #save variables for next step
        P_list[i]=P
        rho_m_i = rho_m
        Z_i = Z
        P_i = P
    
    return P_list

fluid = my_fluid()

pressures = run(my_fluid)

fig = plt.figure()
plt.plot(pressures)
fig.savefig('results/pressureplot.png')