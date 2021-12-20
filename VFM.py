#from _typeshed import Self
import numpy as np
import matplotlib.pyplot as plt
from blackoil import black_oil
from Pipe import Pipe



# --- Variables ---

# input data from tab 2 and 3 (do not need here??)

class TH_model:
    #thermal-hydrailic model based on the drift-flux method
    #input objects: fluid from Black Oil Model, pipe with simple geometry
    #input parameters suggested: inlet liquid flowrate, temperatures and pressure at the outlet
    def __init__(self, fluid, pipe, q_l_in, p_out):
        #TODO: make sure all inputs are valid
        self.fluid = fluid #object of the blackoil class containing functions to handel fluid properties
        self.pipe = pipe    #object of the pipe class containing information about the pipe

        self.q_l_in0 = q_l_in #assume q_l_in is given at standard conditions (else divide by B_l which is dependent on P[0])
        self.p_out = p_out
        self.N = pipe.N #number of gridcells 

        # values for the iteration (might take this values as input in the function instead)
        self.delta = 10**(-5)
        self.maxit = 1000
        self.urf = 0.1 #under relaxation factor

        # Compute superficial oil velocity at standard conditions (found from Timurs code: NewPipe line 169)
        self.j_o0 = (1 - self.fluid.wc) * self.q_l_in0 / self.pipe.D #might have to change q_liq_in


    def massFlux_func(self, rho_l, rho_g, j_l, j_g):
        #denoted G in the paper (eq 9)
        return rho_l*j_l + rho_g*j_g 

    def void_frac_func(self, j, j_g, Cd, Ud):
        #based on the drift-flux model (Zuber and Findlay, 1965) (eq 11)
        return j_g/(Cd*j+Ud)

    def liquid_k_frac(self, j_k, j_sub, void_frac):
        # function to calculate fraction of oil or water (eq 38 and 39)
        # Ex: if calculating oil fraction, input should be (j_o, j_w,...)
        return j_k/(j_k+j_sub)*(1-void_frac)


    #discretization

    def pressureStep(self, i, P_i, rho_m_avg, F_term_avg, horizontal=True):
        # function for finding the pressure uptream P_(i-1) (eq 40)
        # here we set F_term = phi*f_l*G**2/(rho_l*D) for simplicity
        if horizontal:
            return P_i + 0.5*F_term_avg*self.pipe.dx
        else:
            # not properly implemented as pipe do not contain z-vaues
            return P_i + rho_m_avg*9.81*(self.pipe.z[i]-self.pipe.z[i-1]) + 0.5*F_term_avg


    # --- Equations from Andreolli Appendix B (two-phase flow multiplier correlations) ---

    def DarcyFrictionFactor(self, Re):
        # Darcy friction factor f
        f = ((64/Re)**8 + 9.5*(np.log(self.pipe.eps/(3.7*self.pipe.D) + 5.74/(Re**0.9)) - (2500/Re)**6)**(-16))**0.125
        return f

    def VieraGarcia(self, f, f_l, rho, rho_l):
        #return: two-phase multiplier based on correlations proposed by Vierra and Garcia (2014)
        # what friction factor f and density rho is inserted depends on what approach is used 
        # possible approaches are center of volume and center of gravity
        return (f/f_l)*(rho_l/rho)


    def MullerHeck(self, f_l, f_g, rho_l, rho_g, G, x):
        #return: two-phase multiplier based on correlations proposed by Muller-Steinhagen and Heck (1986)
        # not fully implemented as something is missing
        A = 0.5*f_l*G**2/(rho_l*self.pipe.D)
        B = 0.5*f_g*G**2/(rho_g*self.pipe.D)
        Gc = A+ 2*(B-A)*x
        return (Gc*(1-x)**(1/3) + B*x**3)/A

    #TODO: make MONA-function

    def flowrate_k(self, j_k):
        #calculate flowrate for phase k based on superficial velocity
        return j_k*np.pi*self.pipe.D**2 / 4

    def MONA_frictionfactor(self, rho_l, rho_g, void_frac, q_l, q_g, j, visc_l):
        # return two phase multiplier based on 
        # the paper uses different approach for determining the holdup based on slip parameters a1 and a2
        # might have to include the holdup term from Asheim
        a3 = 1.196 # estimated from ecofisk (1.0 for homogenous flow)
        y_ns = q_l/(q_l + q_g)
        rho_ns = rho_l*y_ns + rho_g*(1-y_ns)
        y_l = 1 - void_frac
        visc_ns = visc_l*y_ns + rho_g*(1-y_ns)
        Re_ns = self.pipe.D*j*rho_ns/visc_ns
        f_0 = 0.16/(Re_ns**0.172)
        F = (rho_l*y_ns**2)/(rho_ns*y_l) + rho_g*(1-y_ns)**2/(rho_ns*void_frac)
        return a3*f_0/F

    # --- Void fraction correlations (Equations from Andreolli Appendix C) ---
    def Bendiksen(self, j):
        Fr_j = j/(np.sqrt(9.81*self.pipe.D))
        if Fr_j < 3.5:
            Cd = 1.05 #...+ 0.15*np.sin(angle)
            Ud = np.sqrt(9.81*self.pipe.D)*0.54 #np.sqrt(g_s*D)*(0.35*np.sin(angle) + 0.54*np.cos(angle))
        else: #Fr_j >= 3.5
            Cd = 1.2
            Ud = 0 #0.35*np.sqrt(9.81*self.pipe.D)*np.sin(angle)
        return Cd, Ud

    def woldesemayat_ghajar(self, P, j, j_g, j_l, rho_g, rho_l, R_so): #might split to three funcs
        # Drift flux correlations based on woldesmayat and ghajar
        # return: Cd, Ud
        # ---------------------------------------
        # based on woldesemayat and Gayar (2007)
        # for a horizontal pipe! (may expand later)
        surface_tension = self.fluid.get_surface_tension(R_so)
        Cd = (j_g/j) * (1 + (j_l/j_g)**((rho_g/rho_l)**0.1))
        Ud = 1.0784 * ((1136.214*self.pipe.D*surface_tension*2*(rho_l-rho_g)/rho_l**2))**0.25  # simplified as cos(0)=1 and sin(0)=0
        return Cd, Ud

    # --- Conservation Equations ---

    def compute_mass_balances(self):
        #TODO: make function
        return 0

    def compute_momentum_balances(self):
        #TODO: make function
        return 0


    # --- Andreolli algorithm ----

    def Andreolli_terms(self, P_i, void_frac_method="Bendiksen", twophase_ff_method="VG"):
        # function for finding the density rho_i and frictional term F_term = phi*f_l*G**2/(rho_l*D) for given P (usually P_pred)
        # T is not used as it is isothermal, so T is implemented in the black oil properties

        # black-oil correlations, viscosities, and densities (make one function in myfluid-class to compute all?)
        R_so = self.fluid.R_so_func(P_i)
        B_o = self.fluid.B_o_func(P_i, R_so)
        B_w = self.fluid.B_w_func(P_i)
        Z_g = self.fluid.Z_g_func(P_i)
        R_sl = self.fluid.R_so_func(R_so)
        B_l = self.fluid.B_l_func(B_w, B_o)
        
        #returned from mass eqs
        rho_w = self.fluid.rho_w_func(B_w)
        rho_g = self.fluid.rho_g_func(P_i, Z_g) 
        rho_l = self.fluid.rho_l_func(R_sl, B_l) 
        rho_o = self.fluid.rho_o_func(R_so, B_o)
        
        visc_g = self.fluid.visc_g_func(rho_g) 
        visc_o = self.fluid.visc_o_func(P_i, R_so)
        visc_w = self.fluid.visc_w_func(P_i)
        
        B_g = self.fluid.rho_g0/rho_g      # gas formation volume factor (function?)
        
        # find superficial velocities 
        j_g = B_g*self.j_o0*(self.fluid.GOR-R_so)    #TODO: change to local GOR (if local GOR should be used??)
        j_o = self.j_o0*B_o
        j_w = self.j_o0*self.fluid.get_local_WOR(B_w, B_o)*B_w
        j_l = j_o + j_w
        j   = j_l + j_g
        
        # void and liquid fractions
        if void_frac_method =="Bendiksen":
            Cd, Ud = self.Bendiksen(j)
        else: #woldesmayat and Gayar (currently bug for large N caused by P>P_pb.. =>.. R_so=GOR.. =>.. j_g=0)
            Cd, Ud = self.woldesemayat_ghajar(P_i, j, j_g, j_l, rho_g, rho_l, R_so) #woldesmayat and Gayar
        void_frac = self.void_frac_func(j, j_g, Cd, Ud) #*
        w_frac = self.liquid_k_frac(j_w, j_o, void_frac)
        o_frac = self.liquid_k_frac(j_o, j_w, void_frac)
        
        # additional mixed properties (density and viscosity)
        rho_m = rho_g*void_frac + rho_o*o_frac + rho_w*w_frac
        visc_l = (visc_o*o_frac + visc_w*w_frac)/(o_frac + w_frac) #TODO: include in paper and double check if this is correct according to assumption
        visc_m = visc_g*void_frac + visc_o*o_frac + visc_w*w_frac
        
        G = rho_l*j_l + rho_g*j_g   #total mass flux
        
        # find friction factors        
        Re_l = G*self.pipe.D/visc_l
        f_l = self.DarcyFrictionFactor(Re_l)
        
        # find two phase multiplier (Vierra and Garcia with center of mass)
        #v_m = (rho_g*j_g + rho_o*j_o + rho_w*j_w)/(rho_g*void_frac + rho_o*o_frac + rho_w*w_frac) #double check this eq
        #v_m = (rho_o*j_o + rho_w*j_w)/(rho_o*o_frac + rho_w*w_frac)
        if twophase_ff_method == "VG":
            v_m = j_l/(o_frac + w_frac) #TODO: double check if this is correct
            Re_cm = rho_m*v_m*self.pipe.D/visc_m
            f = self.DarcyFrictionFactor(Re_cm)
            phi = self.VieraGarcia(f, f_l, rho_m, rho_l)
        else: #two-phase friction factor based on the MONA model by Henrik Asheim
            q_l = self.flowrate_k(j_l)
            q_g = self.flowrate_k(j_g)
            phi = self.MONA_frictionfactor(rho_l, rho_g, void_frac, q_l, q_g, j, visc_l)

        F_term = phi*f_l*G**2/(rho_l*self.pipe.D)

        return rho_m, F_term

    #--------------------------------------------------------------------------              

    # Algorithm for silving the problem
    def Andreolli_algorithhm(self, void_frac_method = "Bendiksen", twophase_ff_method="VG"):
        print("algorithm started with void fraction method", void_frac_method, "and two phase friction factor", twophase_ff_method)
        # uses backward differenses and an iterative approach to estimate the new pressure for each backward step based on averages. 
        # Uses P_pred to estimate correlations at new step P_{i-1} and the averages of other properties 
        # --------------------------------------------------------
        # returns: P_list (list with the pressure throug the pipe)

        P_list = np.zeros(self.N) # list for saving the pressure at each step
        P_list[-1] = self.p_out
        # begin with the initial values at the platform
        P_i = self.p_out
        
        # find values for rho_m and F_term at the output
        rho_m_i, F_term_i = self.Andreolli_terms(self.p_out, void_frac_method=void_frac_method, twophase_ff_method=twophase_ff_method)

        # find upstream pressure (iterate backwards)
        for i in range(self.N-1,0,-1): 
            # define initial guess P_(i-1)^pred=P_i
            P_pred=P_i #initialize P-pred to the pressure at upstream grid cell
            error = 1  #initialize epsilon to start iteration
            it=0 # make sure the loop is not eternal
            while error>self.delta:
            
                rho_m, F_term = self.Andreolli_terms(P_pred, void_frac_method = void_frac_method, twophase_ff_method=twophase_ff_method)

                # compute averages
                rho_m_avg = 0.5*(rho_m+rho_m_i) #ajust to 0.5*(rho_m+rho_m_pred) (not sure how to do it for the first step)
                F_term_avg = 0.5*(F_term + F_term_i) 

                #find pressure P_{i-1} from previous pressure and average values of friction term
                P = self.pressureStep(i, P_i, rho_m_avg, F_term_avg)
                
                # P iteration
                # compute error
                error = abs(P_pred-P)/P
                
                #update predicted pressure to new pressure with under-relaxed value TODO: try different urf, and be able to change them
                P_pred += 0.25*(P-P_pred)
                
                #make sure the loop is not eternal
                it+=1
                if it>self.maxit:
                    print("max iterations reached. not convergent for P at step", i)
                    return P_list
            
            #save variables for next step
            P_list[i-1]=P
            rho_m_i = rho_m
            F_term_i = F_term
            P_i = P
            if i%(self.N//10)==0:
                print("iteration at:", i)
        print("iteration ended")
        return P_list

# move all this in main insted
"""
# initialize objects: fluid, pipe, TH_model
fluid = black_oil(50.0, 0.30, 50.0e5, 20.0, 867.0, 0.997, 1020.0, 0.799, T_r = 297.15)
pipe = Pipe(0.2, 0.005, 1000, 100000)
model = TH_model(fluid, pipe, 0.157726, 10e5)

#run algorithm
#pressures = model.run()
pressures = model.Andreolli_algorithhm()
pressures = pressures[1:] #an error so that the last step is not changed TODO: fix error and remove this
#print("pressure at point 10 is:",pressures[10]) #just for testing something
#pressures = run(fluid)
# ------- plotting ----------
# -- plot the pressure through the pipe --
fig = plt.figure()
plt.plot(pressures)
fig.savefig('results/pressureplot.png')
"""
