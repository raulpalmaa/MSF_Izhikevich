###################################################
#############  MSF: IZHIKEVICH MODEL ##############
###########   With Electrical Coupling   ##########
##############   github: raulpalmaa   #############
###################################################
###################################################
# References:
# (1) Master stability function of networks of Izhikevich neuron, Aristides and Cerdeira. (2024)
# (2) Lyapunov exponents computation for hybrid neurons, Bizarri et al. (2013)
# (3) Determining Lyapunov exponents from a time series, Wolf et al. (1985)
# (4) http://csc.ucdavis.edu/~chaos/courses/nlp/Software/partH.html 
#
# Note: I strongly recommend checking these works for a deeper understanding :)
###################################################
# Description:
# This script uses scipy's 'solve_ivp' to integrate the synchronous solution 
# of the Izhikevich neuron model while accounting for electrical coupling. 
# Spikes, which are discontinuities, are handled as events, ensuring the 
# accurate calculation of Lyapunov exponents.
###################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp #https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

def proj(U, V):
    """
    Computes the projection of vector V onto vector U.

    Parameters:
        U (array-like): The vector onto which V is being projected.
        V (array-like): The vector being projected.

    Returns:
        array-like: The projection of V onto U.
    """
    return ((U @ V) / (U @ U)) * U

def gsn(M):
	"""
	Applies the Gram-Schmidt process to orthogonalize the columns of the input matrix.

	Parameters:
		matrix (np.ndarray): A 2D array where each column is a vector to be orthogonalized.

	Returns:
		np.ndarray: A 2D array with orthogonalized (and optionally normalized) columns.
	"""
	v = M
	u = np.zeros((v.shape[0],v.shape[1]))
	u[:,0] = v[:,0] 
	for i in np.arange(1,v.shape[1]):
		p = 0
		for j in range(i):
			p+= proj(u[:,j],v[:,i])  
		u[:,i] = v[:,i] - p
	return u

def append_results(time, v, u, sol):
    """
    Append results to time, v, u, and lys arrays.
    """
    return (
        np.concatenate([time, sol.t[1:]]),
        np.concatenate([v, sol.y[0, 1:]]),
        np.concatenate([u, sol.y[1, 1:]])
    )

def IzkFull(t,ext_state): 
	'''
	Solve the Izhikevich model + Variational Equation.
  Here we consider electrical coupling. Both fv and the Jacobian need to be changed for chemical coupling (see (1)).
	Parameters:
		t (float): Time variable (not used in the equations but required by ODE solvers).
		ext_state (np.ndarray): Extended state vector [v, u, Phi_11, Phi_21, Phi_12, Phi_22].
		sigma (float): Parameter sigma for variational equations.    
	Returns:
		np.ndarray: Derivatives [fv, fu, fPhi_11, fPhi_21, fPhi_12, fPhi_22].
	'''
	v, u = ext_state[:2]  # Membrane potential (v) and recovery variable (u)    
	Phi = ext_state[2:].reshape((2,2))  # State transition matrix

	fv = 0.04 * v**2 + 5 * v + 140 - u + I  # Voltage derivative
	fu = A * (B * v - u)                    # Recovery variable derivative

	JAC = np.array([  # Jacobian matrix
		[(0.08 * v + 5 - sigma), -1],
		[A * B, -A]
	])

	Phi_dot = JAC @ Phi     
	return np.concatenate(([fv, fu], Phi_dot.flatten()))

def S(xym,xyp): 
	"""
	Evaluate the saltation matrix given the state before and after the discontinuity (spike).
	Parameters:
		xym: State variables before the spike.
		xyp: State variables after the spike.    
	Returns:
		np.ndarray: Saltation matrix (Salt).
	"""
	v1, u1 = xym  #x- 
	v2, u2 = xyp  #x+
	Salt = np.zeros((2,2))
	Salt = np.array([[ (v2 / v1)        , 0 ],  #1st row
	                 [ (u2 - u1) / v1   , 1 ]]) #2nd row
	return Salt

def spike(t, y): return y[0]-thre #The event we want to track: Spikes! #check the solve_ivp documents for more info on how 'events' work.

spike.terminal = True
spike.direction = 1


Nd = 2	# Dimension of your system
thre = 30 # Spike: V= 30mV

##Izhikevich ODEs parameters taken from 
##https://doi.org/10.1371/journal.pone.0138919  
##Periodic dynamics
#I = 10
#C = -65
#D = 8
#A = 0.02
#B = 0.2

##Chaotic dynamics
I = -99
C = -56
D = -16
A = 0.2
B = 2

Sg = np.linspace(0.0,0.5,1) # Loop through coupling strenghts. If the coupling is zero then we evaluate the Lyapunov exponent of the isolated system.
MLE = np.zeros(2)
for s in Sg:
    """
    solve_ivp parameters: tspan, IC.
    """
    T0 = 0
    TFINAL = 5000
    IC  = [-60, -110, 1, 0, 0, 1]
    nIterates = 2500 #At each iterate, we evaluate the streching/contraction of the vectors due to the Jacobian - see Fig. 2 of (2).
    U = np.identity(Nd)
    dt = TFINAL/nIterates
    tf = T0+dt
    tspan = [T0, tf]
    dU = np.ones(Nd)
    time = np.array([])
    lys = np.zeros(2)
    v = np.array([])
    u = np.array([])
    LE = np.zeros(Nd)
    LET = np.zeros(Nd)
    GO = True
    sigma = s
    while GO:
        sol = solve_ivp(IzkFull, tspan, IC, method='BDF', events=[spike], atol=1e-7, rtol=1e-4)
        if (sol.status == 0): # no spike!
            if (sol.t[-1]<TFINAL):
                time, v, u = append_results(time, v, u, sol)
                e = sol.y[2:,-1].reshape((2,2))   
                U = e @ U
                U = gsn(U)
                U_norm = np.linalg.norm(U,axis = 0)                
                U /= U_norm  # Normalize columns of U (broadcasting handles division)
    
                LE += np.log(np.abs(U_norm)) 
                LET = LE/sol.t[-1]
                
                lys = np.vstack([lys, LET])
                IC = np.array([sol.y[0,-1], sol.y[1,-1], 1, 0, 0, 1])
                tf = sol.t[-1]+dt
                tspan = [sol.t[-1], tf]
            else:
                GO = False
        elif (sol.status == 1): # spike detected!
            time, v, u = append_results(time, v, u, sol)
            e = sol.y[2:,-1].reshape((2,2))
            
            IC = np.concatenate(([C, sol.y[1,-1] + D], e.flatten()))                    
            Fm = IzkFull(0,sol.y[:,-1])#right before reset
            Fp = IzkFull(0,IC)         #right after reset
            e =  S(Fm[0:2],Fp[0:2]) @ e#saltation matrix applied
            IC = np.concatenate(([C, sol.y[1,-1] + D], e.flatten()))
            tspan = [sol.t[-1], tf]
    MLE = np.vstack((MLE,LET))

MLE = np.delete(MLE, 0, axis = 0) #erasing first row (zeros)

#Plotting results:
plt.plot(time,v,"r-",alpha=0.8,label="v")
plt.plot(time,u,"b-",alpha=0.8,label="u")
plt.show()	

plt.axhline(LET[0],color="black",ls="--",alpha = 0.8)
plt.axhline(LET[1],color="black",ls="--",alpha = 0.8)
plt.plot(lys[:,0],"r-",alpha=0.8)
plt.plot(lys[:,1],"b-",alpha=0.8)
plt.show()	
