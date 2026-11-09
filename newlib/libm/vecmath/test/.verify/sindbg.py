import numpy as np, mpmath as mp
f32=np.float32
def F(x): return np.float32(x)
def fma(a,b,c): return np.float32(np.float64(a)*np.float64(b)+np.float64(c))
def mul(a,b): return np.float32(np.float64(a)*np.float64(b))
def add(a,b): return np.float32(np.float64(a)+np.float64(b))
def sub(a,b): return np.float32(np.float64(a)-np.float64(b))
S1=F(-1.6666667163e-01);S2=F(8.3333337680e-03);S3=F(-1.9841270114e-04)
S4=F(2.7557314297e-06);S5=F(-2.5050759689e-08);S6=F(1.5896910177e-10)
u=F(0.7835541367530823); e=F(-4.59629454496735e-08)
mp.mp.dps=60
U=mp.mpf(float(u)); E=mp.mpf(float(e)); T=U+E
def ulp_of(v):
    if v==0: return mp.mpf(2)**-149
    return mp.ldexp(mp.mpf(1), int(mp.floor(mp.log(abs(v),2)))-23)
# exact sin
S=mp.sin(T); print("exact sin",float(S))
zh=mul(u,u); zl=fma(u,u,-zh)
v=mul(zh,u); ve=fma(zh,u,-v); ve=fma(zl,u,ve)
R=fma(zh,S6,F(S5)); R=fma(zh,R,F(S4)); R=fma(zh,R,F(S3)); R=fma(zh,R,F(S2)); R=fma(zh,R,F(S1))
C=mul(v,R); Ce=fma(v,R,-C); Ce=fma(ve,R,Ce)
print("R",repr(float(R)),"exact R~", float((S-T)/ (T**3)))
# exact C = sin(T)-T? C = sin-T (higher terms) = sin(T) - (u+e)
Cexact = S - T
print("C",repr(float(C)),"Cexact",float(Cexact),"C err",float(C-Cexact),"in ulp of sin",float((mp.mpf(float(C))-Cexact)/ulp_of(S)))
Ce_tot = mp.mpf(float(C))+mp.mpf(float(Ce))
print("C+Ce err vs Cexact",float((Ce_tot-Cexact)/ulp_of(S)))
print("zl",float(zl),"ve",float(ve))
print("v err exact:",float((mp.mpf(float(v))+mp.mpf(float(ve))-U**3)/ulp_of(S)))
# What sin would be with exact C: 
print("u+e+Cexact err", float((T+Cexact-S)/ulp_of(S)))
# check R evaluation error vs exact R using mp
mp.dps=60
zhex=mp.mpf(float(zh))
Rex=mp.mpf('-0.16666666666666666666666666666666666666666666666667') # not right approach
# compute R exact with Taylor coefficients at z=U^2
Z=U*U
Rex=sum(mp.mpf(1)/mp.factorial(2*k+3)*mp.power(-1,k)*mp.power(Z,k) for k in range(6))
print("Rex",float(Rex),"R-Rex",float(mp.mpf(float(R))-Rex),"rel",float((mp.mpf(float(R))-Rex)/abs(Rex)))
print("v*R vs Cexact",float((mp.mpf(float(v))*mp.mpf(float(R))-Cexact)/ulp_of(S)))
