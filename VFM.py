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

        self.q_l_in = q_l_in
        self.p_out = p_out
        self.N = pipe.N #number of gridcells 

        # values for the iteration (might take this values as input in the function instead)
        self.delta = 10**(-4)
        self.maxit = 1000
        self.urf = 0.1 #under relaxation factor

        # Compute superficial oil velocity at standard conditions (found from Timurs code: NewPipe line 169)
        self.j_o0 = (1 - self.fluid.wc) * self.q_l_in / self.pipe.D #might have to change q_liq_in

        
    """
    GOR =196.4          #gas-oil ratio (obtained from black oil object)
    WOR =15.6           #water-oil ratio (obtain from black oil object)
    P_plat =22.2*10**5  # pressure at platform (p_out)(given as input above)
    P_WCT =104.6*10**5  # pressure at WCT (used for testing)
    j_o0 = 0.6860       # oil superficial velocity at SC (get from ...???)
    eps = 0.00021       # roughness (from pipe)
    D = 0.1524          # pipe diameter (from pipe)

    g_s = 9.81      #gravity acceleration (do not need?)
    # ---

    theta = 0 #angle relative to the horizontal plane (do not need for now)

    N = 10000 #(get from pipe)
    #z = np.zeros(22) #insert from tab 1
    x = np.linspace(0, 1000, N) #(from pipe)

    delta=10**(-4)
    maxit = 100
    #define under-relaxation factor
    urf = 0.1
    """

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
            return P_i + 0.5*F_term_avg
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
        # not fully implemented as 
        A = 0.5*f_l*G**2/(rho_l*self.pipe.D)
        B = 0.5*f_g*G**2/(rho_g*self.pipe.D)
        Gc = A+ 2*(B-A)*x
        return (Gc*(1-x)**(1/3) + B*x**3)/A

    # --- Equations from Andreolli Appendix C (Void fraction correlations) ---
    def Bendiksen(self, j):
        Fr_j = j/(np.sqrt(9.81*self.pipe.D))
        if Fr_j < 3.5:
            Cd = 1.05 #...+ 0.15*np.sin(angle)
            Ud = np.sqrt(9.81*self.pipe.D)*0.54 #np.sqrt(g_s*D)*(0.35*np.sin(angle) + 0.54*np.cos(angle))
        else: #Fr_j >= 3.5
            Cd = 1.2
            Ud = 0 #0.35*np.sqrt(9.81*self.pipe.D)*np.sin(angle)
        return Cd, Ud

    def Cd_and_Ud_func(self, P, j, j_g, j_l, rho_g, rho_l, R_so): #might split to three funcs
        # Drift flux correlations
        # TODO: implement Woldesemayat and Ghajar(2007) instead? use surface tension as in timurs code
        # return: Cd, Ud
        # ---------------------------------------
        Cd, Ud = 0, 0
        # default method Bendiksen(1984)
        """
        Fr_j = j/(np.sqrt(g_s*D))
        if Fr_j < 3.5:
            Cd = 1.05 + 0.15*np.sin(angle)
            Ud = np.sqrt(g_s*D)*(0.35*np.sin(angle) + 0.54*np.cos(angle))
        else: #Fr_j >= 3.5
            Cd = 12
            Ud = 0.35*np.sqrt(g_s*D)*np.sin(angle)
        """

        # badsed on woldesemayat and Gayar (2007)
        # for a horizontal pipe!
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


    ###################################################################
    #               Andreolli algorithm
    ###################################################################

    def Andreolli_terms(self, P_i, void_frac_method="Bendiksen"):
        # function for finding the density rho_i and frictional term F_term = phi*f_l*G**2/(rho_l*D) for given P (usually P_pred)
        # T is not used as it is isothermal, so T is implemented in the black oil properties
        # black-oil correlations, viscosities, and densities (make one function in myfluid-class to compute all?)
        R_so = fluid.R_so_func(P_i)
        B_o = fluid.B_o_func(P_i, R_so)
        B_w = fluid.B_w_func(P_i)
        Z_g = fluid.Z_g_func(P_i)
        R_sl = fluid.R_so_func(R_so)
        B_l = fluid.B_l_func(B_w, B_o)
        
        #returned from mass eqs
        #rho_w = fluid.rho_w0/B_w    #function? *
        rho_w = fluid.rho_w_func(B_w)
        rho_g = fluid.rho_g_func(P_i, Z_g) #*
        rho_l = fluid.rho_l_func(R_sl, B_l) #*
        rho_o = fluid.rho_o_func(R_so, B_o)
        
        visc_g = fluid.visc_g_func(rho_g) #**
        visc_o = fluid.visc_o_func(P_i, R_so)
        visc_w = fluid.visc_w_func(P_i)
        
        B_g = fluid.rho_g0/rho_g      # gas formation volume factor (function?)
        
        # find superficial velocities **
        j_g = B_g*self.j_o0*(fluid.GOR-R_so)    #TODO: change to local GOR (if local GOR should be used??)
        j_o = self.j_o0*B_o
        #j_w = self.j_o0*fluid.WOR*B_w   #TODO: change to local WOR (done below)
        j_w = self.j_o0*fluid.get_local_WOR(B_w, B_o)*B_w
        j_l = j_o + j_w
        j   = j_l + j_g
        
        # void and liquid fractions
        if void_frac_method=="Bendiksen":
            Cd, Ud = self.Bendiksen(j)
        else: #woldesmayat and Gayar (currently bug for large N caused by P>P_pb.. =>.. R_so=GOR.. =>.. j_g=0)
            Cd, Ud = self.Cd_and_Ud_func(P_i, j, j_g, j_l, rho_g, rho_l, R_so) #woldesmayat and Gayar
        void_frac = self.void_frac_func(j, j_g, Cd, Ud) #*
        w_frac = self.liquid_k_frac(j_w, j_o, void_frac)
        o_frac = self.liquid_k_frac(j_o, j_w, void_frac)
        
        # additional mixed properties (density and viscosity)
        rho_m = rho_g*void_frac + rho_o*o_frac + rho_w*w_frac
        visc_l = (visc_o*o_frac + visc_w*w_frac)/(o_frac + w_frac) #double check if this is correct according to assumption
        visc_m = visc_g*void_frac + visc_o*o_frac + visc_w*w_frac
        
        G = rho_l*j_l + rho_g*j_g   #total mass flux
        
        # find friction factors        
        Re_l = G*self.pipe.D/visc_l
        f_l = self.DarcyFrictionFactor(Re_l)
        
        # find two phase multiplier (Vierra and Garcia with center of mass)
        #v_m = (rho_g*j_g + rho_o*j_o + rho_w*j_w)/(rho_g*void_frac + rho_o*o_frac + rho_w*w_frac) #double check this eq
        #v_m = (rho_o*j_o + rho_w*j_w)/(rho_o*o_frac + rho_w*w_frac)
        v_m = j_l/(o_frac + w_frac)
        Re_cm = rho_m*v_m*self.pipe.D/visc_m
        f = self.DarcyFrictionFactor(Re_cm)
        phi = self.VieraGarcia(f, f_l, rho_m, rho_l)

        F_term = phi*f_l*G**2/(rho_l*self.pipe.D)

        return rho_m, F_term


    # ----------------------------------------------------------------------
    """
    #the new algorithm which is wrong. Atempted to mix andreolli with "other" approach, and now it makes no sense
    def run_test(self, void_frac_method="Bendiksen"):
        p_out = self.p_out
        p_in = 3.5*p_out #arbitrary guessed value at beginning of pipe
        #TODO: optimize guessed value for p_in with more qualified guess

        #P_list = np.linspace(p_in, p_out, self.N) # list for saving the pressure at each step, linear guess for now
        #P_pred = P_list
        P_pred = np.linspace(p_in, p_out, self.N)
        iteration = 0   # initialize number of iterations
        max_error = 1     # initilize error (epsilon) larger than self.delta
        rho_m_i = 0     # need better way to do this.. (needed at fist iteration beacuse of boundary)
        F_term_i = 0 # .. and this..
        error_list =[]

        while max_error>=self.delta:
            #check if max iterations are reached
            P_list = P_pred
            if iteration>self.maxit:
                print("max iterations reached. not convergent for P at step", iteration)
                error_array = np.array(error_list)
                return P_list, error_array
            for i in range(self.N-2,0,-1):  #find upstream pressure (iterate backwards)
                #* if computation is done in 
                
                # black-oil correlations, viscosities, and densities (make one function in myfluid-class to compute all?)
                R_so = fluid.R_so_func(P_list[i])
                B_o = fluid.B_o_func(P_list[i], R_so)
                B_w = fluid.B_w_func(P_list[i])
                Z_g = fluid.Z_g_func(P_list[i])
                R_sl = fluid.R_so_func(R_so)
                B_l = fluid.B_l_func(B_w, B_o)
                
                #returned from mass eqs
                rho_w = fluid.rho_w0/B_w    #function? *
                rho_g = fluid.rho_g_func(P_list[i], Z_g) #*
                rho_l = fluid.rho_l_func(R_sl, B_l) #*
                rho_o = fluid.rho_o_func(R_so, B_o)
                
                visc_g = fluid.visc_g_func(rho_g) #**
                visc_o = fluid.visc_o_func(P_list[i], R_so)
                visc_w = fluid.visc_w_func(P_list[i])
                
                B_g = fluid.rho_g0/rho_g      # gas formation volume factor (function?)
                
                # find superficial velocities **
                j_g = B_g*self.j_o0*(fluid.GOR-R_so)    #TODO: change to local GOR
                #print(j_g)
                #exit()
                j_o = self.j_o0*B_o
                j_w = self.j_o0*fluid.WOR*B_w   #TODO: change to local WOR
                j_l = j_o + j_w
                j   = j_l + j_g
                
                # void and liquid fractions
                #surface_tension = fluid.get_surface_tension(R_so)
                if void_frac_method=="Bendiksen":
                    Cd, Ud = self.Bendiksen(j)
                else: #woldesmayat and gayar (currently bug caused by P>P_pb.. =>.. R_so=GOR.. =>.. j_g=0)
                    Cd, Ud = self.Cd_and_Ud_func(P_list[i], j, j_g, j_l, rho_g, rho_l, R_so) #Bendiksen (have not implemented the others)
                void_frac = self.void_frac_func(j, j_g, Cd, Ud) #*
                w_frac = self.liquid_k_frac(j_w, j_o, void_frac)
                o_frac = self.liquid_k_frac(j_o, j_w, void_frac)
                
                # additional mixed properties (density and viscosity)
                rho_m = rho_g*void_frac + rho_o*o_frac + rho_w*w_frac
                visc_l = (visc_o*o_frac + visc_w*w_frac)/(o_frac + w_frac) #double check
                visc_m = visc_g*void_frac + visc_o*o_frac + visc_w*w_frac
                
                G = rho_l*j_l + rho_g*j_g   #total mass flux
                
                # find friction factors        
                Re_l = G*self.pipe.D/visc_l
                f_l = self.DarcyFrictionFactor(Re_l)
                
                # find two phase multiplier (Vierra and Garcia or Muller and Heck)
                # "VGm"(default), center of mass
                #v_m = (rho_g*j_g + rho_o*j_o + rho_w*j_w)/(rho_g*void_frac + rho_o*o_frac + rho_w*w_frac) #double check this eq
                #v_m = (rho_o*j_o + rho_w*j_w)/(rho_o*o_frac + rho_w*w_frac)
                v_m = j_l/(o_frac + w_frac)
                rho = rho_m
                Re_cm = rho_m*v_m*self.pipe.D/visc_m
                f = self.DarcyFrictionFactor(Re_cm)
                phi = self.VieraGarcia(f, f_l, rho, rho_l)
           
                # compute averages
                if i<self.N-1: #after first step
                    rho_m_avg = 0.5*(rho_m+rho_m_i) #ajust to 0.5*(rho_m+rho_m_pred) (not sure how to do it for the first step)
                
                    F_term=phi*f_l*G**2/(rho_l*self.pipe.D)
                    F_term_avg = 0.5*(F_term + F_term_i) 
                    
                else:   # initial step (do not need)
                    rho_m_avg = rho_m #ajust to 0.5*(rho_m+rho_m_pred) (not sure how to do it for the first step)
                    
                    F_term=phi*f_l*G**2/(rho_l*self.pipe.D)
                    F_term_avg = F_term #ajust 

                                #find pressure
                P = self.pressureStep(i, P_list[i+1], rho_m_avg, F_term_avg)

                #save variables for next step
                P_list[i]=P
                rho_m_i = rho_m
                F_term_i = F_term
                
            # P iteration
            # check convergence (compute epsilon)
            relative_error = abs(P_pred-P_list)/P_list
            max_error = np.max(relative_error)
            error_list.append(max_error)
            # under relaxed predicted value
            #check covergence (epsilon)
            P_pred += 0.5*(P_list-P_pred)

            
            #make sure the loop is not eternal
            iteration+=1
            if(iteration % 100 == 0):
                print("max error at iteration", iteration, "is: ", max_error)
                print("pressure at point 10 is:",P_list[10])
            #print(max_error)
        print("Convergence after", iteration, "iterations")
        error_array = np.array(error_list)
        return P_list, error_array

    # ----------------------------------------------------------------------------------------------------              


    #old algorithm. Actually correct according to the Andreolli code
    # uses backward differenses and an iterative approach to estimate the new pressure for each backward step. 
    # Uses P_pred to estimate correlations at new step P_{i-1} and the averages of 
    def run(self, void_frac_method = "Bendiksen"):
        
        P_list = np.zeros(self.N) # list for saving the pressure at each step
        P_list[-1] =self.p_out
        #begin with the initial values at the platform
        P_i = self.p_out
        
        #initialize some values
        rho_m_i = 0 
        F_term_i = 0
        #find values for rho_m and Z at the output
        # black-oil correlations, viscosities, and densities (make one function in myfluid-class to compute all?)
        R_so = fluid.R_so_func(P_i)
        B_o = fluid.B_o_func(P_i, R_so)
        B_w = fluid.B_w_func(P_i)
        Z_g = fluid.Z_g_func(P_i)
        R_sl = fluid.R_so_func(R_so)
        B_l = fluid.B_l_func(B_w, B_o)
        
        #returned from mass eqs
        rho_w = fluid.rho_w0/B_w    #function? *
        rho_g = fluid.rho_g_func(P_i, Z_g) #*
        rho_l = fluid.rho_l_func(R_sl, B_l) #*
        rho_o = fluid.rho_o_func(R_so, B_o)
        
        visc_g = fluid.visc_g_func(rho_g) #**
        visc_o = fluid.visc_o_func(P_i, R_so)
        visc_w = fluid.visc_w_func(P_i)
        
        B_g = fluid.rho_g0/rho_g      # gas formation volume factor (function?)
        
        # find superficial velocities **
        j_g = B_g*self.j_o0*(fluid.GOR-R_so)    #TODO: change to local GOR
        #print(j_g)
        #exit()
        j_o = self.j_o0*B_o
        j_w = self.j_o0*fluid.WOR*B_w   #TODO: change to local WOR
        j_l = j_o + j_w
        j   = j_l + j_g
        
        # void and liquid fractions
        #surface_tension = fluid.get_surface_tension(R_so)
        if void_frac_method=="Bendiksen":
            Cd, Ud = self.Bendiksen(j)
        else: #woldesmayat and gayar (currently bug caused by P>P_pb.. =>.. R_so=GOR.. =>.. j_g=0)
            Cd, Ud = self.Cd_and_Ud_func(P_i, j, j_g, j_l, rho_g, rho_l, R_so) #Bendiksen (have not implemented the others)
        void_frac = self.void_frac_func(j, j_g, Cd, Ud) #*
        w_frac = self.liquid_k_frac(j_w, j_o, void_frac)
        o_frac = self.liquid_k_frac(j_o, j_w, void_frac)
        
        # additional mixed properties (density and viscosity)
        rho_m = rho_g*void_frac + rho_o*o_frac + rho_w*w_frac
        visc_l = (visc_o*o_frac + visc_w*w_frac)/(o_frac + w_frac) #double check
        visc_m = visc_g*void_frac + visc_o*o_frac + visc_w*w_frac
        
        G = rho_l*j_l + rho_g*j_g   #total mass flux
        
        # find friction factors        
        Re_l = G*self.pipe.D/visc_l
        f_l = self.DarcyFrictionFactor(Re_l)
        
        # find two phase multiplier (Vierra and Garcia or Muller and Heck)
        # "VGm"(default), center of mass
        #v_m = (rho_g*j_g + rho_o*j_o + rho_w*j_w)/(rho_g*void_frac + rho_o*o_frac + rho_w*w_frac) #double check this eq
        #v_m = (rho_o*j_o + rho_w*j_w)/(rho_o*o_frac + rho_w*w_frac)
        v_m = j_l/(o_frac + w_frac)
        #rho = rho_m
        Re_cm = rho_m*v_m*self.pipe.D/visc_m
        f = self.DarcyFrictionFactor(Re_cm)
        phi = self.VieraGarcia(f, f_l, rho_m, rho_l)

        rho_m_i = rho_m # not needed line..
        F_term_i=phi*f_l*G**2/(rho_l*self.pipe.D)

        for i in range(self.N-1,0,-1):  #find upstream pressure (iterate backwards)
            #--define initial guess P_(i-1)^pred=P_i
            P_pred=P_i
            error = 10  #initialize epsilon to start iteration
            it=0 # make sure the loop is not eternal
            while error>self.delta:
            
                # black-oil correlations, viscosities, and densities (make one function in myfluid-class to compute all?)
                R_so = fluid.R_so_func(P_pred)
                B_o = fluid.B_o_func(P_pred, R_so)
                B_w = fluid.B_w_func(P_pred)
                Z_g = fluid.Z_g_func(P_pred)
                R_sl = fluid.R_so_func(R_so)
                B_l = fluid.B_l_func(B_w, B_o)
                
                #returned from mass eqs
                rho_w = fluid.rho_w0/B_w    #function? *
                rho_g = fluid.rho_g_func(P_pred, Z_g) #*
                rho_l = fluid.rho_l_func(R_sl, B_l) #*
                rho_o = fluid.rho_o_func(R_so, B_o)
                
                visc_g = fluid.visc_g_func(rho_g) #**
                visc_o = fluid.visc_o_func(P_pred, R_so)
                visc_w = fluid.visc_w_func(P_pred)
                
                B_g = fluid.rho_g0/rho_g      # gas formation volume factor (function?)
                
                # find superficial velocities **
                j_g = B_g*self.j_o0*(fluid.GOR-R_so)    #TODO: change to local GOR
                #print(j_g)
                #exit()
                j_o = self.j_o0*B_o
                j_w = self.j_o0*fluid.WOR*B_w   #TODO: change to local WOR
                j_l = j_o + j_w
                j   = j_l + j_g
                
                # void and liquid fractions
                #surface_tension = fluid.get_surface_tension(R_so)
                if void_frac_method=="Bendiksen":
                    Cd, Ud = self.Bendiksen(j)
                else: #woldesmayat and gayar (currently bug caused by P>P_pb.. =>.. R_so=GOR.. =>.. j_g=0)
                    Cd, Ud = self.Cd_and_Ud_func(P_pred, j, j_g, j_l, rho_g, rho_l, R_so) 
                void_frac = self.void_frac_func(j, j_g, Cd, Ud) #*
                w_frac = self.liquid_k_frac(j_w, j_o, void_frac)
                o_frac = self.liquid_k_frac(j_o, j_w, void_frac)
                
                # additional mixed properties (density and viscosity)
                rho_m = rho_g*void_frac + rho_o*o_frac + rho_w*w_frac
                visc_l = (visc_o*o_frac + visc_w*w_frac)/(o_frac + w_frac) #double check
                visc_m = visc_g*void_frac + visc_o*o_frac + visc_w*w_frac
                
                G = rho_l*j_l + rho_g*j_g   #total mass flux
                
                # find friction factors        
                Re_l = G*self.pipe.D/visc_l
                f_l = self.DarcyFrictionFactor(Re_l)
                
                # find two phase multiplier (Vierra and Garcia or Muller and Heck)
                # "VGm"(default), center of mass
                #v_m = (rho_g*j_g + rho_o*j_o + rho_w*j_w)/(rho_g*void_frac + rho_o*o_frac + rho_w*w_frac) #double check this eq
                #v_m = (rho_o*j_o + rho_w*j_w)/(rho_o*o_frac + rho_w*w_frac)
                v_m = j_l/(o_frac + w_frac)
                #rho = rho_m
                Re_cm = rho_m*v_m*self.pipe.D/visc_m
                f = self.DarcyFrictionFactor(Re_cm)
                phi = self.VieraGarcia(f, f_l, rho_m, rho_l)
           
                # compute averages
                #if i<self.N-1: #after first step
                rho_m_avg = 0.5*(rho_m+rho_m_i) #ajust to 0.5*(rho_m+rho_m_pred) (not sure how to do it for the first step)
            
                F_term=phi*f_l*G**2/(rho_l*self.pipe.D)
                F_term_avg = 0.5*(F_term + F_term_i) 

                #find pressure P_{i-1}
                P = self.pressureStep(i, P_i, rho_m_avg, F_term_avg)
                
                # P iteration
                # check convergence (compute epsilon)
                error = abs(P_pred-P)/P
                
                # under relaxed predicted value
                #check covergence (epsilon)
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
        return P_list
    """
    # ----------------------------------------------------------------------------------------------------              


    #old algorithm. Actually correct according to the Andreolli code
    # uses backward differenses and an iterative approach to estimate the new pressure for each backward step. 
    # Uses P_pred to estimate correlations at new step P_{i-1} and the averages of 
    def Andreolli_algorithhm(self, void_frac_method = "Bendiksen"):

        P_list = np.zeros(self.N) # list for saving the pressure at each step
        P_list[-1] =self.p_out
        #begin with the initial values at the platform
        P_i = self.p_out
        
        #find values for rho_m and F_term at the output
        rho_m_i, F_term_i = self.Andreolli_terms(self.p_out, void_frac_method=void_frac_method)

        for i in range(self.N-1,0,-1):  #find upstream pressure (iterate backwards)
            #--define initial guess P_(i-1)^pred=P_i
            P_pred=P_i #initialize P-pred to the pressure at upstream grid cell
            error = 1  #initialize epsilon to start iteration
            it=0 # make sure the loop is not eternal
            while error>self.delta:
            
                rho_m, F_term = self.Andreolli_terms(P_pred, void_frac_method= void_frac_method)

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
        return P_list

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

exit() #this is wrong (error is )
# -- plot the max error --
fig = plt.figure()
plt.plot(errors)
plt.yscale('log')
fig.savefig('results/maxerrors.png')
