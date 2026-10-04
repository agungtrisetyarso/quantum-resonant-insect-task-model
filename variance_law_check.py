import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm, gamma
rng=np.random.default_rng(1)
n_list=[1,2,4]
def colony(theta,n,xs):
    # find s* with mean P = xs (kappa=0)
    f=lambda ls: np.mean(1/(1+np.exp(-n*(ls-np.log(theta)))))-xs
    ls=brentq(f,-50,50); s=np.exp(ls)
    P=1/(1+np.exp(-n*(ls-np.log(theta))))
    # numeric derivative s dF/ds
    h=1e-6
    Fp=np.mean(1/(1+np.exp(-n*(ls+h-np.log(theta)))));Fm=np.mean(1/(1+np.exp(-n*(ls-h-np.log(theta)))))
    E=(Fp-Fm)/(2*h)
    return s,P,E
maxerr=0
for trial in range(200):
    n=rng.choice(n_list); xs=rng.uniform(0.1,0.9)
    kind=rng.integers(3)
    N=200
    if kind==0: th=np.exp(rng.normal(0,rng.uniform(0.05,2),N))
    elif kind==1: th=rng.gamma(1/rng.uniform(0.05,1)**2,1,N)
    else: th=np.where(rng.random(N)<0.5,1,np.exp(rng.uniform(0,3)))
    s,P,E=colony(th,n,xs)
    pred=n*(xs*(1-xs)-P.var())
    maxerr=max(maxerr,abs(E-pred))
print("variance law max abs err",maxerr)
# probit kernel: log-concave -> E <= n psi(x*)
mx=-1
for trial in range(200):
    n=2;xs=rng.uniform(0.1,0.9);th=np.exp(rng.normal(0,rng.uniform(0.05,2),100))
    f=lambda ls: np.mean(norm.cdf(n*(ls-np.log(th))))-xs
    ls=brentq(f,-50,50); E=np.mean(n*norm.pdf(n*(ls-np.log(th))))
    mx=max(mx,E-n*norm.pdf(norm.ppf(xs)))
print("probit: max(E - n psi(x*)) (should be <=0):",mx)
