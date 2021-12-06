import numpy as np
"""
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

gamma_g = gamma_g0 # from eq. 30

"""
class black_oil:
    #the black oil model. Initial imput values from tab 2 and 3 (changed now)
    # change input to GOR, water cut, p_bp, t_bp, standard density of oil, gas and water and gas specific gravity to calculate the fluid properties
    # things that are no longer constants: WOR?, GOR?
    # not sure what to do with T (and also C3 which is related to T at WCT) (for now: T_r as input)
    # all equation numbers correspond to the Andreolli paper
    def __init__(self, GOR, wc, p_bp, t_bp, rho_o0, rho_g0, rho_w0, gamma_g0 = 0.00, T_r = 333):
        
        self.wc = wc            #water cut
        self.WOR = wc / (1 - wc)  # water-oil ratio
        self.gamma_g0 = gamma_g0     #gas specific gravity at SC
        self.T_r = T_r           #reservoir temperature
        self.P_b = p_bp         # bubble pressure at T_r (constant as the system is isoterm)
        self.GOR = GOR          #gas-oil ratio
        self.T_k =T_r           # temperature at WCT 

        self.rho_o0 = rho_o0    # oil density at sc
        self.rho_g0 = rho_g0    # gas density at sc
        self.rho_w0 = rho_w0    # water density at sc

        self.gamma_g = gamma_g0 # from eq. 30
        self.gamma_o0 = self.rho_o0 / self.rho_w0  # specific gravity of oil (from Timurs code)(eq 23)
        self.API = 141.5 / self.gamma_o0 - 131.5         #API gravity (denoted gamma_API in tab)(eq 22)

        # -- from Appendix A (black oil correlations) ---
        # input from tab A.5 (black oil model)

        self.C_gas = 8.314   # universial gas constant (denoted /\ )
        self.M_a = 0.028966  #Air molar mass

        # constants to ensure correct units (si-units)
        self.C1 = 5.61456349 
        self.C2 = 0.000145038
        self.C3 = 1.8*self.T_k-459.67
        self.C4 = 0.001
        self.C5 = 1.8

        # --- define some more values at SC ---

        self.rho_l0 = (self.WOR/(1+self.WOR))*self.rho_w0 + (1/(1+self.WOR))*self.rho_o0 #liquid density at SC (eq 26)

        self.P_atm = 100325       #NEED GLOBAL VARIABLE?

    def get_local_WOR(self, B_w, B_o):
        return self.WOR * (B_w / B_o)


    def get_local_wc(self, WOR_local):
        return WOR_local / (1 + WOR_local)


    def R_so_func(self, P):
        #solution gas_oil ratio (eq A.1)
        if P>self.P_b:
            return self.GOR
        #else: #P<=P_b
        return self.gamma_g/self.C1*((self.C2*P/18.2 + 1.4) * 10**(0.0125*self.API-0.00091*self.C3) )**1.2048


    def B_o_func(self, P, R_so):
        #oil formation volume factor (eq A.2+)
        B_o = 0.9759 + 0.00012*(self.C1*R_so*(self.gamma_g0/self.gamma_o0)**0.5 + 1.25*self.C3)**1.2
        if P <= self.P_b:
            return B_o
        else:
            c0 = (-1433 + 5*self.C1*self.GOR + 17.2*self.C3 - 1180*self.gamma_g0 + 12.61*self.API)/(P*10**5)
            return B_o*np.exp(-c0*(P-self.P_b))


    def B_w_func(self, P):
        # water formation volume factor (eq A.5+)
        k1 = 5.50654*10**(-7) *(self.C3)**2
        k2 = -1.72834*10**(-13) *self.C2**2 *self.C3*P**2
        k3 = -3.58922*10**(-7) *self.C2*P
        k4 = -2.25341*10**(-10) *self.C2**2 *P**2
        dV_wT = -1.0001*10**(-2) + 1.33391*10**(-4)*self.C3 + k1
        dV_wP = -1.95301*10**(-9)*self.C2*self.C3*P + k2 + k3 + k4
        return (1+dV_wP)*(1+dV_wT)


    def Z_g_func(self, P):
        #gas compressibility factor    (eq A.12+)
        # calculate pseudo critical pressure and temperature
        P_pc = (677 + 15.0*self.gamma_g - 37.5*self.gamma_g**2)/self.C2
        T_pc = (168 + 325*self.gamma_g - 12.5*self.gamma_g**2)/self.C5
        
        P_pr=P/P_pc
        T_pr=self.T_k/T_pc
        return 1 - (3.52*P_pr/(10**(0.9813*T_pr))) + (0.274*P_pr**2/(10**(0.8157*T_pr)))

    def visc_o_func(self, P, R_so):
        #oil viscosity (eq A.17+)
        m1 = 10**(0.43 + 8.33/self.API)
        u_od = self.C4*(0.32 + (1.8*10**7)/(self.API**4.53))*(360/(self.C3+200))**m1 
        if P <= self.P_atm:    #P==P_atm in paper
            return u_od 
        else:
            k5 = (u_od/self.C4)**(5.44*(self.C1*R_so+150)**(-0.338))
            u_o = 10.715*self.C4*(self.C1*R_so + 100)**(-0.515)*k5
            if P <= self.P_b:
                return u_o
            #else
            k6 = -11.513 - 8.98*10**(-5)*self.C2*P
            m2 = 2.6*(self.C2*P)**1.187*np.exp(k6)
            return u_o*(P/self.P_b)**m2


    def visc_g_func(self, rho_g):
        #gas viscosity
        M_g = self.gamma_g*self.M_a #gas molar mass
        F1 = ((9.379+16.07*M_g)*(self.C5*self.T_k)**1.5)/(209.2+19260*M_g+self.C5*self.T_k)
        F2 = 3.448 + 986.4/(self.C5*self.T_k) + 10.09*M_g
        F3 = 2.447 - 0.2224*F2
        return self.C4*F1*10**(-4)*np.exp(F2*(self.C4*rho_g)**F3)


    def visc_w_func(self, P):
        #water wiscosity
        u_w0 = 109.574*self.C4*self.C3**(-1.12166)
        k7 = self.C2*P + 14.7
        return u_w0*(0.9994 + 4.0295*10**(-5)*k7 + 3.1062*10**(-9)*k7**2)


    def R_sl_func(self, R_so):
        #solution gas-liquid(?) ratio (eq 17)
        return (1/(1+self.WOR))*R_so


    def B_l_func(self, B_w, B_o):
        # liquid formation volume factor (eq 21)
        #TODO: change to use local WOR??
        return (self.WOR/(1+self.WOR))*B_w + (1/(1+self.WOR))*B_o


    def B_g_func(self, rho_g):
        return self.rho_g0/rho_g


    def rho_g_func(self, P, Z_g):
        # gas density   (eq 28)
        return self.gamma_g*self.M_a*P/(self.C_gas*self.T_r*Z_g)


    def rho_o_func(self, R_so, B_o):
        # density oil  (eq 24)
        return (self.rho_o0 + self.rho_g0*R_so)/B_o


    def rho_l0_func(self, rho_l, WOR):
        # density liquid at sc
        #TODO: bestem deg for input og hva som skal brukes innebygd
        return 0 #temporary


    def rho_l_func(self, R_sl, B_l):
        # density liquid (eq 25)
        return (self.rho_g0*R_sl+self.rho_l0)/B_l #bytte ut rho_l0


    def get_surface_tension(self, R_so):
        #TODO: ask riz about this!!
        """
        Gas-oil surface tension correlation adapted from paper Abdul-Majeed
        and Abu Al-Soof "Estimation of gas–oil surface tension".
        Returns:
            - surface tension, [N/m]
        """
        # function from timur. Not checked against paper
        a = 1.11591 - 0.00305 * (self.T_r -273.15)  # temperature switched from Kelvin to Celsius
        sigma_do = a * (38.084 - 0.259 * self.API)
        # Compute live oil (at local conditions) surface tension
        if R_so < 50:
            sigma_lo = sigma_do * (11 + 0.02549 * R_so**1.0157)
        else:
            sigma_lo = sigma_do * 32.0436 * (R_so**(-1.1367))
        return 0.001 * sigma_lo


