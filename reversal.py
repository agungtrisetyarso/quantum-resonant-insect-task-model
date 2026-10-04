import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
def zeta(sig,n,kap,A=4.0,xs=0.5,N=4000):
    th=np.exp(sig*norm.ppf((np.arange(N)+0.5)/N))
    F=lambda s,x: np.mean(s**n/(s**n+(th*(1+kap*x))**n))
    s=brentq(lambda s: F(s,xs)-xs,1e-6,1e6); h=1e-6
    Fs=(F(s+h,xs)-F(s-h,xs))/(2*h); Fx=(F(s,xs+h)-F(s,xs-h))/(2*h)
    J=np.array([[0,-A],[Fs,Fx-1]])  # tau=1, alpha=A
    ev=np.linalg.eigvals(J); w0=np.sqrt(np.prod(ev).real); g=-ev.sum().real/2
    P=s**n/(s**n+(th*(1+kap*xs))**n); H=P.var()/(xs*(1-xs))
    return g/w0,-Fx,H
for n,kap in [(2,1),(4,0),(4,1),(4,4),(8,4)]:
    out=[zeta(s,n,kap) for s in [0.02,0.3,0.6,1.0]]
    print(n,kap," ".join(f"[H={H:.2f} eta={e:.2f} z={z:.3f}]" for z,e,H in out))
