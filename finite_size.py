"""Finite-size check of the step response (Table 7): agent-based ensemble mean vs
(i) large-N mean-field equations and (ii) the O(1/N) system-size correction
(second-order van Kampen / effective mesoscopic rate equations), kappa=0."""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from scipy.integrate import solve_ivp
A=4.0; n=2; sig=0.2; tau=1.0; T=30.0; dt=0.02
def thetas(N): return np.exp(sig*norm.ppf((np.arange(N)+0.5)/N))
def P(s,th): return s**n/(s**n+th**n)
def eq_s(th,xs): return brentq(lambda s: P(s,th).mean()-xs,1e-6,1e6)
def overshoot(t,x,x0,x1,smooth=11):
    xm=np.convolve(x,np.ones(smooth)/smooth,mode='same')
    plateau=x[t>=T-5].mean(); core=(t>1)&(t<T-5)
    return 100*(xm[core].max()-plateau)/(plateau-x0)
def abm(N,ncol,seed=1):
    rng=np.random.default_rng(seed); th=thetas(N); s0=eq_s(th,0.5)
    a=rng.random((ncol,N))<P(s0,th); s=np.full(ncol,s0); steps=int(T/dt)
    xs=np.empty(steps); ts=np.arange(steps)*dt
    for k in range(steps):
        x=a.mean(1); xs[k]=x.mean()
        s=np.maximum(s+(0.6*A-A*x)*dt,1e-12)
        re=rng.random((ncol,N))<dt
        a=np.where(re,rng.random((ncol,N))<P(s[:,None],th),a)
    return overshoot(ts,xs,0.5,0.6)
def expansion(N,order):
    th=thetas(N); s0=eq_s(th,0.5); p0=P(s0,th)
    def F(s): return P(s,th).mean()
    def Fs(s): h=1e-5*s; return (F(s+h)-F(s-h))/(2*h)
    def Fss(s): h=1e-4*s; return (F(s+h)-2*F(s)+F(s-h))/h**2
    # state: ms, mx, p_i (N), Sss, Ssx, Sxx
    def rhs(t,y):
        ms,mx=y[0],y[1]; p=y[2:2+N]; Sss,Ssx,Sxx=y[2+N:]
        Pi=P(ms,th); corr=0.5*Fss(ms)*Sss if order==1 else 0.0
        dms=0.6*A-A*mx; dmx=(F(ms)+corr-mx)/tau; dp=(Pi-p)/tau
        J=np.array([[0,-A],[Fs(ms)/tau,-1/tau]])
        Dxx=np.mean(Pi*(1-p)+p*(1-Pi))/(N*tau)
        S=np.array([[Sss,Ssx],[Ssx,Sxx]]); dS=J@S+S@J.T+np.diag([0,Dxx])
        return np.r_[dms,dmx,dp,dS[0,0],dS[0,1],dS[1,1]]
    y0=np.r_[s0,p0.mean(),p0,0.0,0.0,np.mean(p0*(1-p0))/N]
    ts=np.arange(0,T,dt); sol=solve_ivp(rhs,(0,T),y0,t_eval=ts,rtol=1e-8,atol=1e-10)
    return overshoot(ts,sol.y[1],0.5,0.6)
if __name__=="__main__":
    print(" N   ABM    mean-field   1/N-corrected")
    for N,nc in [(8,20000),(16,20000),(64,5000),(256,1500),(1024,400)]:
        print(f"{N:5d} {abm(N,nc):6.1f} {expansion(N,0):8.1f} {expansion(N,1):10.1f}",flush=True)
