class blackoil:

    def __init__(self, P):
        P.self = P

    def R_so_func(P):
        #solution gas_oil ratio (eq A.1)
        if P>P_b:
            return GOR
        #else: #P<=P_b
            return gamma_g/C1*((C2*P/18.2 + 1.4) * 10**(0.0125*API-0.00091*C3) )**1.2048
        
    def B_o_func(P, R_so):
        #oil formation volume factor (eq A.2+)
        B_o = 0.9759 + 0.00012*(C1*R_so*(gamma_g0/gamma_o0)**0.5 + 1.25*C3)**1.2
        if P <= P_b:
            return B_o
        else:
            c0 = (-1433 + 5*C1*GOR + 17.2*C3 - 1180*gamma_g0 + 12.61*API)/(P*10**5)
            return B_o*np.exp(-c0*(P-P_b))

    def B_w_func(P):
        # water formation volume factor (eq A.5+)
        k1 = 5.50654*10**(-7) *(C3)**2
        k2 = -1.72834*10**(-13) *C2**2 *C3*P**2
        k3 = -3.58922*10**(-7) *C2*P
        k4 = -2.25341*10**(-10) *C2**2 *P**2
        dV_wT = -1.0001*10**(-2) + 1.33391*10**(-4)*C3 + k1
        dV_wP = -1.95301*10**(-9)*C2*C3*P + k2 + k3 + k4
        return (1+dV_wP)*(1+dV_wT)

    def Z_g_func(P):
        #gas compressibility factor    (eq A.12+)
        # calculate pseudo critical pressure and temperature
        P_pc = (677 + 15.0*gamma_g - 37.5*gamma_g**2)/C2
        T_pc = (168 + 325*gamma_g - 12.5*gamma_g**2)/C5
        
        P_pr=P/P_pc
        T_pr=T_k/T_pc
        return 1 - (3.52*P_pr/(10**(0.9813*T_pr))) + (0.274*P_pr**2/(10**(0.8157*T_pr)))

    def visc_o_func(P, R_so):
        #oil viscosity (eq A.17+)
        m1 = 10**(0.43 + 8.33/API)
        u_od = C4*(0.32 + (1.8*10**7)/(API**4.53))*(360/(C3+200))**m1 
        if P <= P_atm:    #P==P_atm in paper
            return u_od 
        else:
            k5 = (u_od/C4)**(5.44*(C1*R_so+150)**(-0.338))
            u_o = 10.715*C4*(C1*R_so + 100)**(-0.515)*k5
            if P <= P_b:
                return u_o
            #else
            k6 = -11.513 - 8.98*10**(-5)*C2*P
            m2 = 2.6*(C2*P)**1.187*np.exp(k6)
            return u_o*(P/P_b)**m2
        
    def visc_g_func(rho_g):
        #gas viscosity
        M_g = gamma_g*M_a #gas molar mass
        F1 = ((9.379+16.07*M_g)*(C5*T_k)**1.5)/(209.2+19260*M_g+C5*T_k)
        F2 = 3.448 + 986.4/(C5*T_k) + 10.09*M_g
        F3 = 2.447 - 0.2224*F2
        return C4*F1*10**(-4)*np.exp(F2*(C4*rho_g)**F3)

    def visc_w_func(P):
        #water wiscosity
        u_w0 = 109.574*C4*C3**(-1.12166)
        k7 = C2*P + 14.7
        return u_w0*(0.9994 + 4.0295*10**(-5)*k7 + 3.1062*10**(-9)*k7**2)

