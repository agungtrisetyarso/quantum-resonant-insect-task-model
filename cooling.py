"""Stimulus and workforce variance with passive stimulus decay mu (ds/dt = delta - alpha x - mu s).
Checks the closed-form Lyapunov solution against (i) a numerical Lyapunov solve and (ii) agent-based colonies."""
import numpy as np
from scipy.linalg import solve_continuous_lyapunov
from scipy.stats import norm
def theory(N,H,xs,ss,alpha,tau,n,mu,eta=0.0):
    Q=xs*(1-xs)*(1-H); Fs=n*Q/ss; B=2*Q/(N*tau)
    a,b,c,d=mu,alpha,Fs/tau,(1+eta)/tau
    Vs=b*b*B/(2*(a+d)*(b*c+a*d)); Vx=B*(b*c+a*(a+d))/(2*(a+d)*(b*c+a*d))
    J=np.array([[-a,-b],[c,-d]]); S=solve_continuous_lyapunov(J,-np.diag([0,B]))
    assert np.allclose(S,[[Vs,-a/b*Vs],[-a/b*Vs,Vx]],rtol=1e-9), S
    return Vs,Vx
def abm(N,sig,A=2.0,n=2,mu=0.5,T=2500,dt=0.02,ncol=24,seed=5):
    rng=np.random.default_rng(seed); th=np.exp(sig*norm.ppf((np.arange(N)+0.5)/N))
    delta=A*0.5+mu*1.0   # puts x*=1/2 at s*=1 for log-symmetric thresholds
    a=rng.random((ncol,N))<0.5; s=np.ones(ncol); b=int(100/dt)
    cnt=0
    for k in range(int(T/dt)):
        x=a.mean(1); s=np.maximum(s+(delta-A*x-mu*s)*dt,1e-9)
        re=rng.random((ncol,N))<dt
        P=s[:,None]**n/(s[:,None]**n+th**n)
        a=np.where(re,rng.random((ncol,N))<P,a)
        if k>=b:
            if cnt==0: sx=np.zeros(ncol);sxx=np.zeros(ncol);s1=np.zeros(ncol);s2=np.zeros(ncol)
            sx+=x;sxx+=x*x;s1+=s;s2+=s*s;cnt+=1
    vx=np.mean(sxx/cnt-(sx/cnt)**2); vs=np.mean(s2/cnt-(s1/cnt)**2)
    P=1/(1+th**-n); H=P.var()/0.25
    return H,vx,vs
if __name__=="__main__":
    N=64; A=2.0; mu=0.5
    c=mu*1.0/(A*2*0.25); print("cooling number c =",c)
    rows=[]
    for sig in [0.05,0.8,1.6]:
        H,vx,vs=abm(N,sig,mu=mu)
        Vs,Vx=theory(N,H,0.5,1.0,A,1.0,2,mu)
        Vs0,_=theory(N,H,0.5,1.0,A,1.0,2,0.0)
        rows.append((sig,H,vs,Vs,vx,Vx,Vs0))
        print(f"sigma={sig} H={H:.3f}  Var(s) ABM={vs:.5f} theory={Vs:.5f} | Var(x) ABM={vx:.5f} theory={Vx:.5f} | mu=0 Var(s)={Vs0:.5f}")
    H0=rows[0][1]
    for r in rows: print("ratio Var(s)/Var(s)_hom ABM %.3f theory %.3f formula %.3f"%(r[2]/rows[0][2],r[3]/rows[0][3],(1+c/(1-H0))/(1+c/(1-r[1]))))
