import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.stats import norm
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
C=["#2a78d6","#eb6834","#1baf7a"]; M=["o","s","^"]
rng=np.random.default_rng(7)
# (a) exact variance law across random colonies
pts={0:[],1:[],2:[]}
for t in range(240):
    n=rng.choice([1,2,4,8]); xs=rng.uniform(0.1,0.9); k=t%3; N=200
    if k==0: th=np.exp(rng.normal(0,rng.uniform(0.05,2),N))
    elif k==1: th=rng.gamma(1/rng.uniform(0.1,1.2)**2,1,N)
    else: th=np.where(rng.random(N)<0.5,1,np.exp(rng.uniform(0,3)))
    L=np.log(th)
    ls=brentq(lambda z: np.mean(1/(1+np.exp(-n*(z-L))))-xs,-60,60)
    P=1/(1+np.exp(-n*(ls-L))); h=1e-6
    E=(np.mean(1/(1+np.exp(-n*(ls+h-L))))-np.mean(1/(1+np.exp(-n*(ls-h-L)))))/(2*h)
    H=P.var()/(xs*(1-xs)); pts[k].append((1-H,E/(n*xs*(1-xs))))
# (b) agent-based check of the scaling triad (vectorised over colonies)
def abm(N,sig,A=2.0,n=2,T=2500,dt=0.02,ncol=24,seed=3):
    rng=np.random.default_rng(seed); th=np.exp(sig*norm.ppf((np.arange(N)+0.5)/N))
    a=rng.random((ncol,N))<0.5; s=np.ones(ncol); b=int(100/dt); steps=int(T/dt)
    S1x=S2x=S1s=S2s=0.0; cnt=0
    for k in range(steps):
        x=a.mean(1); s=np.maximum(s+(0.5*A-A*x)*dt,1e-9)
        re=rng.random((ncol,N))<dt
        P=s[:,None]**n/(s[:,None]**n+th**n)
        a=np.where(re,rng.random((ncol,N))<P,a)
        if k>=b:
            S1x+=x; S2x+=x*x; S1s+=s; S2s+=s*s; cnt+=1
    vx=np.mean(S2x/cnt-(S1x/cnt)**2); vs=np.mean(S2s/cnt-(S1s/cnt)**2)
    P=1/(1+th**-n); H=P.var()/0.25
    return H,vx,A**2*vx/vs,vs
sigs=[0.05,0.4,0.8,1.2,1.6]; res=[abm(64,s) for s in sigs]
H0,vx0,w0,_=res[0]
fig,ax=plt.subplots(1,3,figsize=(7.4,2.7))
lab=["log-normal","gamma","two-type"]
for k in range(3):
    p=np.array(pts[k]); ax[0].scatter(p[:,0],p[:,1],s=14,marker=M[k],facecolor="none",edgecolor=C[k],lw=0.9,label=lab[k])
ax[0].plot([0,1],[0,1],color="#555",lw=1,zorder=0)
ax[0].set_xlabel(r"$1-H$"); ax[0].set_ylabel(r"$E\,/\,[n\,x^*(1-x^*)]$")
ax[0].set_title("(a) Variance law, 240 colonies",loc="left",fontsize=9); ax[0].legend(frameon=False,fontsize=8)
ax[0].set_xlim(0,1.02); ax[0].set_ylim(0,1.02)
r=np.array([[(1-h)/(1-H0),vx/vx0,w/w0] for h,vx,w,_ in res])
ax[1].plot([0.2,1.02],[0.2,1.02],color="#555",lw=1,zorder=0,label="theory")
ax[1].scatter(r[:,0],r[:,1],s=36,marker="o",color=C[0],label=r"$\mathrm{Var}(x)$, ABM",zorder=3)
ax[1].scatter(r[:,0],r[:,2],s=36,marker="s",facecolor="none",edgecolor=C[1],lw=1.3,label=r"$\omega_0^2$, ABM",zorder=3)
ax[1].set_xlabel(r"$(1-H)/(1-H_0)$"); ax[1].set_ylabel("ratio to most homogeneous")
ax[1].set_title(r"(b) Scaling triad, $N=64$",loc="left",fontsize=9); ax[1].legend(frameon=False,fontsize=7.5,loc="upper left")
hs=np.array([r_[0] for r_ in res]); vss=np.array([r_[3] for r_ in res])
ax[2].axhline(1,color="#555",lw=1,zorder=0,label="theory (any $H$)")
ax[2].scatter(hs,vss*64*2/(2.0*1*1),s=36,marker="D",color=C[2],zorder=3,label="ABM")
ax[2].set_ylim(0.8,1.2); ax[2].set_xlim(-0.03,0.7)
ax[2].set_xlabel(r"$H$"); ax[2].set_ylabel(r"$\mathrm{Var}(s)\,Nn/(\alpha\tau s^*)$")
ax[2].set_title("(c) Stimulus-noise invariance",loc="left",fontsize=9); ax[2].legend(frameon=False,fontsize=7.5,loc="upper left")
ax[0].set_title("(a) Variance law",loc="left",fontsize=9)
fig.tight_layout(); fig.savefig("fig8_variance_law.pdf"); fig.savefig("fig8_variance_law.png",dpi=160)
for (h,vx,w,vs),s in zip(res,sigs): print(f"sigma={s} H={h:.3f} Var={vx:.5f} th={0.25*(1-h)/64:.5f} w0^2={w:.4f} th={2*2*0.25*(1-h):.4f} Vs={vs:.5f} th=0.015625")
