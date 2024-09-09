"""
crashback.py: Class implementation of surge equation of motion for ship crashback simulation.

(c) 2024 The Regents of the University of Michigan

This code is licensed under the GNU GPL v3 license, available at https://opensource.org/license/gpl-3-0
"""

import numpy as np
import pandas as pd
import os
import math
from scipy.interpolate import interp1d

__author__ = "Kyle E. Marlantes"
__license__ = "GPLv3"
__version__ = "0.0.0"
__maintainer__ = "Kyle Marlantes"
__email__ = "kylemarl@umich.edu"
__status__ = "Research"

class Crashback ():
    """Implements a nonlinear, second-order ODE model of ship crashback.

    Equation of motion:

    (m + a11)*xdd = T - R

    ...where m is the physical mass, a11 is the added mass in surge, T is the thrust, and R is the resistance.

    xdd is the second derivative of position (acceleration).

    Time integration is optionally 1st or 2nd order implicit, BDF1 or BDF2.

    The equation is discretized as follows:

    v = state vector v[0] = x, v[1] = xd
    vd = derivative of state vector vd[0] = xd, vd[1] = xdd

    A = coefficient matrix = [ 0, 1]
                             [ 0, 0]  

    q = forcing vector q[0] = 0, q[1] = (T-R)/(m+a11)
                   
    Written in standard form:

    vd = A*v + q

    Discretization using implicit finite-differencing:

    BDF1:

    ( v_{n+1} - v_{n} ) / dt = A*v_{n+1} + q_{n+1}

    BDF2:

    ( 3*v_{n+1} - 4*v_{n} + v_{n-1} ) / 2*dt = A*v_{n+1} + q_{n+1}

    ...where n is the discrete step at the current time, and n+1 is the next step in the solution.

    All terms can be collected to the left-hand-side to obtain a discrete equation equal to zero.
    This form can be solved using a root finding method, in this case we use a discrete Newton's method.
    The Jacobian in the Newton update is approximated using a finite-difference with an arbitrary step.

    Additional references:
    [Roddy2006] Robert F. Roddy, David E. Hess, and Will Faller (2006) NEURAL NETWORK PREDICTIONS OF THE 4-QUADRANT WAGENINGEN PROPELLER SERIES
    [Delen2023] Cihad Delen, Sakir Bal (2023) A comprehensive experimental investigation of total drag and wave height of ONR Tumblehome, including uncertainty analysis
    """

    def __init__(self, m, tf, rpm_shape=[], D=0, nProp=1, WSA=0, L=0, a11=0, thrust_data=['SeriesB','B470data.csv'], resistance_data='ITTC1957'):

        """
        m: [kg] physical mass of the vessel

        tf: [s] time required to reverse propeller from n0 to nf

        rpm_shape: [n0, nf] propeller RPM [rotations per minute] n0 is the initial rotation, and nf is the final rotation at tf

        D: [m] propeller diameter

        nProp: [-] number of propellers

        WSA: [m^2] wetted surface area, required for ITTC1957 method

        L: [m] characteristic length (usually length over waterline)

        a11: [kg] added mass in surge
        
        thrust_data: list of two elements
                     first entry is a data identifier:
                         default 'SeriesB' will use the predefined Wageningen B Series data
                         optional 'JKTKQ' will use a user file of three columns, J, KT, and KQ. First line reserved for column headings
                         optional 'BCTCQ' will use a user file of three columns, Beta (in deg), CT, and CQ. First line reserved for column headings
                     second entry is the datafile name.

        resistance_data: default 'ITTC1957' will use the ITTC 1957 skin-friction line
                          or, user specified path and filename to resistance data, must follow format of example file            
        """

        self.thrust_method = 'User'
        if thrust_data[0]=='SeriesB':
            self.thrust_method = thrust_data[0]
            # check for and read in predefined datafiles
            df = pd.read_csv(thrust_data[1], skiprows=0, delim_whitespace=False)
            self.propeller = {}
            self.propeller['K'] = np.array(df['K']) 
            self.propeller['T-A(k)'] = np.array(df['T-A(k)']) 
            self.propeller['T-B(k)'] = np.array(df['T-B(k)'])
        elif thrust_data[0]=='JKTKQ':
            # table of J, KT, and KQ
            self.thrust_method = thrust_data[0]
            df = pd.read_csv(thrust_data[1], skiprows=0, delim_whitespace=True)
            self.propeller = {}
            self.propeller['J'] = np.array(df['J']) 
            self.propeller['K_T'] = np.array(df['K_T']) 
            self.propeller['K_Q'] = np.array(df['K_Q'])
            self.propeller['K_T_interp'] = interp1d(np.array(df['J']), np.array(df['K_T']), kind='linear')
        elif thrust_data[0]=='BCTCQ':
            # table of Beta, CT, and CQ
            self.thrust_method = thrust_data[0]
            df = pd.read_csv(thrust_data[1], skiprows=0, delim_whitespace=True)
            self.propeller = {}
            self.propeller['Beta'] = np.array(df['#Beta']) 
            self.propeller['C_T'] = np.array(df['C_T']) 
            self.propeller['C_Q'] = np.array(df['C_Q'])
            self.propeller['C_T_interp'] = interp1d(np.radians(np.array(df['#Beta'])), np.array(df['C_T']), kind='linear')
        else:
            # self.thrust_method = 'User'
            self.thrust_data_fn = thrust_data
            # read user data and prepare interpolant
            
        if resistance_data=='ITTC1957':
            self.resistance_method = resistance_data
            if WSA==0:
                raise ValueError('WSA must be provided when using ITTC1957 resistance method.')
            if L==0:
                raise ValueError('L must be provided when using ITTC1957 resistance method.')
        else:
            # self.thrust_method = 'User'
            self.resistance_data_fn = resistance_data
            # read user data and prepare interpolant

        self.m = m
        self.tf = tf
        self.rpm_shape = rpm_shape
        self.WSA = WSA
        self.L = L
        self.a11 = a11
        self.D = D
        self.nProp = nProp
        self.nu = 1.1304e-6 # [m^2/s] seawater at 17C per ITTC water properties
        self.rho = 1025.5633 # [kg/m^3]

        self.idx = 0 # make this a private variable? make a "global" index instead of passing n all the time...
    
    def Re (self,u):
        """Returns Reynolds number for given speed."""
        return u*self.L/self.nu

    def coef_to_force (self,C,u):
        return C * 0.5 * self.rho * u**2 * self.WSA
        
    def advance_ratio (self,uA,N):
        """Returns the advance ratio.
        uA: [m/s] speed of advance
        N: rpm"""
        nn = N/60. # convert to rps
        if nn==0:
            return 0
        else:
            return uA/(nn*self.D)

    def compute_beta (self,J):
        """Returns Beta defined in ref. QQQ"""
        beta = math.atan2(J,0.7*np.pi)
        if beta<0: beta+=np.pi # to match sign convention of Roddy2006
        elif beta==0: beta+=0.5*np.pi
        return beta

    def CF_ittc_1957 (self,u):
        """Computes CF using the ITTC 1957 skin friction line."""   
        if u<=0: return 0
        else:
            return 0.075 / ((np.log10(self.Re(u)) - 2.)**2)

    def CT_BSeries (self,beta):
        Ak = self.propeller['T-A(k)']
        Bk = self.propeller['T-B(k)']
        k = self.propeller['K']
        CT = 0.01*np.sum(Ak*np.cos(np.radians(k*np.degrees(beta))) + Bk*np.sin(np.radians(k*np.degrees(beta))))
        return CT

    def compute_rpm (self,t):
        """Returns rpm for current time."""
        if t<self.tf:
            return (self.rpm_shape[1]-self.rpm_shape[0])/self.tf * t + self.rpm_shape[0]
        else:
            return self.rpm_shape[1] # always final rpm
        
    def compute_thrust (self,uA,N):
        """Computes the thrust for the propeller(s).
        uA: [m/s] advance velocity
        N: [rpm] propeller revolutions per minute
        """
        nn = N/60. # rev per second
        J = self.advance_ratio(uA,N)
        beta = self.compute_beta(J)
        if self.thrust_method=='User':
            # use interpolant
            return 0
        elif self.thrust_method=='JKTKQ':
            if J>max(self.propeller['J']): J=max(self.propeller['J']) # overrun of interpolant
            if J<min(self.propeller['J']): J=min(self.propeller['J'])
            KT = self.propeller['K_T_interp'](J)
            return KT * self.rho * abs(nn)**2 * self.D**4
        elif self.thrust_method=='BCTCQ':
            CT = self.propeller['C_T_interp'](beta)
            return CT * 0.5*self.rho*(uA**2 + (0.7*np.pi*abs(nn)*self.D)**2) * 0.25*np.pi*self.D**2
        elif self.thrust_method=='SeriesB':
            # Appendix B in Roddy2006            
            if nn==0.: return 0 # CT_BSeries yields small nonzero values for beta=0, only valid for uA=0, N!=0
            else:
                CT = self.CT_BSeries(beta)
                return CT * 0.5*self.rho*(uA**2 + (0.7*np.pi*abs(nn)*self.D)**2) * 0.25*np.pi*self.D**2

    def compute_resistance (self,u):
        """Computes the resistance for the given speed."""
        if self.resistance_method=='User':
            # use interpolant
            return 0
        elif self.resistance_method=='ITTC1957':
            CF = self.CF_ittc_1957(u)
            return self.coef_to_force(CF,u)
    
    def solve (self,max_time=0,dt=0,tol=0.0001,x0=[0,0],order=2):
        """Solves the equation of motion with BDF2."""

        def compute_q(u_new0,t,n):
            """Compute the forcing vector on the right-hand-side."""  
            rpm = self.compute_rpm(t)
            # passing u_new0[1] below assumes advance velocity is equal to ship velocity            
            # multiply by number of propellers for multi-screw vessels
            T = self.compute_thrust(u_new0[1],rpm)
            R = self.compute_resistance(u_new0[1]) 
            f = self.nProp*T - R
            return np.array([0.,f/(self.m+self.a11)])

        def compute_accel (u_new0,A0,t,n):
            q_k = compute_q(u_new0,t,n)
            return np.dot(A0,u_new0) + q_k 
        
        def compute_G_BDF2 (u_new0,u_now0,u_old0,A0,t,n):             
            accel = compute_accel(u_new0,A0,t,n)
            return (3./(2.*dt))*u_new0 - (4./(2.*dt))*u_now0 + (1./(2.*dt))*u_old0 - accel

        def compute_G_BDF1 (u_new0,u_now0,A0,t,n):  
            accel = compute_accel(u_new0,A0,t,n)
            return (1./dt)*u_new0 - (1./dt)*u_now0 - accel

        def compute_Gk (u_new0,u_now0,u_old0,A0,t,n):
            if order==1: # BDF1
                return compute_G_BDF1 (u_new0,u_now0,A0,t,n)
            elif order==2:
                if n==1: # BDF1 for first step
                    return compute_G_BDF1 (u_new0,u_now0,A0,t,n) 
                else: # BDF2
                    return compute_G_BDF2 (u_new0,u_now0,u_old0,A0,t,n)

        def NewtonStep (u_new0,u_now0,u_old0,A0,t,n):
            """Implicit state update to solve system of nonlinear equations using Newton root-finding technique."""            
            J=np.zeros([Nu,Nu])

            u_new1=u_new0.copy()
            u_now1=u_now0.copy()
            u_old1=u_old0.copy()
                
            err=100.
            inner_loop=0 
            while err>tol: # inner implicit loop
                inner_loop+=1
                G_k0 = compute_Gk(u_new1,u_now1,u_old1,A0,t,n) # unperturbed right-hand-side of system 
                
                # build Jacobian
                u_new00=u_new1.copy()
                du=0.01 # state perturbation for finite difference                
                for i in range(Nu): # for each state variable
                    u_new1[i]+=du # perturb state variable                    
                    G_k = compute_Gk(u_new1,u_now1,u_old1,A0,t,n) # update perturbed right-hand-side of system                   
                    J[:,i] = (G_k - G_k0)/du # finite difference to estimate derivatives
                    u_new1=u_new00.copy() # restore unperturbed state vector

                u_new1=u_new00.copy() # restore unperturbed state vector                
                del_k = np.linalg.solve(-J,G_k0) # solve system            
                u_new1 = u_new1 + del_k # Newton update
                err=np.linalg.norm(del_k,ord=None) # defaults to 2-norm for vectors
                # print('inner_loop',inner_loop,'err',err)
                if inner_loop>10:
                    print('breaking inner loop...')
                    break

            # update state vectors
            u_old1 = u_now1.copy()
            u_now1 = u_new1.copy()

            return u_new1,u_now1,u_old1

        Nu=2

        self.time = [0]
        self.pos,self.vel,self.acc = [0],[0],[0]
        self.rpm,self.beta,self.J = [0],[0],[0]
        self.thrust,self.drag = [0],[0]

        A = np.array([[0,1],[0,0]])

        # state vectors
        u_new,u_now,u_old=np.ones([Nu]),np.ones([Nu]),np.ones([Nu]) # initialize state vectors (initial conditions are zero)

        u_now[0]*=x0[0] # initial position
        u_now[1]*=x0[1] # initial velocity
        self.pos[0] = u_now[0]
        self.vel[0] = u_now[1]
        self.rpm[0] = self.rpm_shape[0]

        # update initial thrust here
        self.thrust[0] = self.nProp*self.compute_thrust(self.vel[0],self.rpm[0])
        self.drag[0] = self.compute_resistance(self.vel[0]) 

        t=self.time[0]; n=0
        while t<=max_time:
            t+=dt; n+=1

            u_new,u_now,u_old=NewtonStep(u_new,u_now,u_old,A,t,n)

            self.time.append(t)
            self.pos.append(u_new[0])
            self.vel.append(u_new[1])
            self.acc.append(compute_accel(u_new,A,t,n)[1])

            self.rpm.append(self.compute_rpm(t)) 
            self.J.append(self.advance_ratio(u_new[1],self.rpm[n]))
            self.beta.append(self.compute_beta(self.J[n]))

            self.thrust.append(self.nProp*self.compute_thrust(u_new[1],self.rpm[n]))
            self.drag.append(self.compute_resistance(u_new[1]))

            if self.beta[n]>=np.radians(179.9): # exit condition prior to backing
                print(r'...exiting on Beta = 180 degrees.')
                break 
            
            if (t%(0.1*max_time)<=1e-2):             
                print('...time ',f'{t:.1f}',' of ',max_time)
