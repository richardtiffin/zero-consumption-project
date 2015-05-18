__author__ = 'aas00jrt'
import numpy as np
import scipy.stats as sp
# TODO the code for the wishart draw comes out of github and should probably be checked
import invwishart as iw
rnorm = np.random.normal
rmvnorm=np.random.multivariate_normal
runif = np.random.rand
zeros = np.zeros
ones=np.ones
dot=np.dot
add=np.add
mean=np.mean
eye = np.identity
transpose = np.transpose
diag = np.diag
shape = np.shape
chol = np.linalg.cholesky
inv=np.linalg.inv
conc = np.concatenate
xp=np.expand_dims
sq=np.squeeze
from scipy.stats import distributions
import matplotlib.pyplot as plt
from math import *
from random import random
import statistics as stat
import scipy
import scipy.linalg
import scipy.stats as sp
from sys import exit

#### Generate data
t = 500
# Covariance

Sigmatrue = np.zeros((3,3))
Sigmatrue[0,0] = 4
Sigmatrue[0,1] = -0.5
Sigmatrue[0,2] = -0.5
Sigmatrue[1,0] = Sigmatrue[0,1]
Sigmatrue[1,1] = 3
Sigmatrue[1,2] = -0.5
Sigmatrue[2,0] = Sigmatrue[0,2]
Sigmatrue[2,1] = Sigmatrue[1,2]
Sigmatrue[2,2] = 2

#True values
d1t = 0.5
d2t = 0.5
gt=3
b1t=4
b2t=5
gibbsno=10000
z1=xp(rnorm(0,2,t),axis=1)
z2=xp(rnorm(0,2,t),axis=1)
w=ones((t,1))

#starting values
phi=100*eye(3)
d1 = 2
d2 = 3
g=3
b1=4
b2=5
om2=1
om3=1
#bdraw=conc((conc((g,b1)),b2))
bdraw=(g,b1,b2)
bdraw=xp(bdraw,1)
#storage arrays
storeb=zeros((gibbsno,3))
stored=zeros((gibbsno,2))
stores=zeros((gibbsno,9))

l=chol(Sigmatrue)
e=rnorm(0,1,(t,3))
e=dot(l,e.T).T
x1=z1*d1t+xp(e[:,0],1)
x2=z2*d2t+xp(e[:,1],1)
y=w*gt+x1*b1t+x2*b2t+xp(e[:,2],1)
temp=conc((w,x1),1)
xmat = conc((temp,x2),1)
shapex=shape(xmat)
nobs=shapex[0]

# Sigmatrue = np.matrix(np.zeros((4,4)))
# Sigmatrue[0,0] = 2
# Sigmatrue[0,1] = 0.5
# Sigmatrue[0,2] = 0.3
# Sigmatrue[0,3] = -0.2
# Sigmatrue[1,0] = Sigmatrue[0,1]
# Sigmatrue[1,1] = 3
# Sigmatrue[1,2] = 0.2
# Sigmatrue[1,3] = 0.1
# Sigmatrue[2,0] = Sigmatrue[0,2]
# Sigmatrue[2,1] = Sigmatrue[1,2]
# Sigmatrue[2,2] = 1.5
# Sigmatrue[2,3] = -0.3
# Sigmatrue[3,0] = Sigmatrue[0,3]
# Sigmatrue[3,1] = Sigmatrue[1,3]
# Sigmatrue[3,2] = Sigmatrue[2,3]
# Sigmatrue[3,3] = 1.3

def ldl(a):
    n=(shape(a)[0])
    l=eye(n)
    d=zeros((n,n))
    for i in range(0,n):
        did = diag(d)
        if i > 0:
            if i==1:
                lint=l[i,0]*l[i,0]
                dint=d[0,0]
            else:
                lint=l[i,0:i]*l[i,0:i]
                dint=did[0:i]
            ldint=np.dot(transpose(lint),dint)
        else:
            ldint=0
        d[i,i]=a[i,i]-ldint
        for j in range(i+1,n):
            if i > 0:
                if i==1:
                    lint=l[j,0]*l[i,0]
                    lint=dot((lint),did[0])
                else:
                    lint=l[j,0:i]*l[i,0:i]
                    lint=dot((lint),did[0:i])/did[i]
            else:
                lint=0
            l[j,i]=(a[j,i]-lint)/did[i]
    return(l,d)


for i in range(0,gibbsno):
    print("loop",i)
    #sigdraw=iw.invwishartrand(t-1,phi)
    e1=x1-z1*d1
    e2=x2-z2*d2
    e3=y-dot(xmat,bdraw)
    alle=conc((e1,e2,e3),1)
    phi=dot(alle.T,alle)
    s1=dot(e1.T,e1)
    om1=sp.invgamma.rvs(t,scale=s1)
    p21bar=dot(inv(s1),dot(e1.T,e2))
    p21var=om2*inv(s1)
    p21draw=rnorm(p21bar,p21var)
    u2=e2-e1*p21draw
    print(mean(u2))
    s2=dot(u2.T,u2)
    om2=sp.invgamma.rvs(t,scale=s2)
    e1e2=conc((e1,e2),1)
    is3=inv(dot(e1e2.T,e1e2))
    p31p32bar=dot(is3,dot(e1e2.T,e3))
    p31p32bar=sq(p31p32bar)
    p31p32var=om3*is3
    p31p32draw=xp(rmvnorm(p31p32bar,p31p32var),1)
    u3=e3-dot(e1e2,p31p32draw)
    print(mean(u3))
    s3=dot(u3.T,u3)
    om3=sp.invgamma.rvs(t,scale=s3)
    p31draw=p31p32draw[0]
    p32draw=p31p32draw[1]
    ia=((1,0,0),(-p21draw,1,0),(-p31draw,-p32draw,1))
    a=inv(ia)
    h=((om1,0,0),(0,om2,0),(0,0,om3))
    sigdraw=dot(a,dot(h,a.T))
    stores[i,:]=np.reshape(sigdraw,(1,9))

    e12=conc((e1.T,e2.T),axis=0)
    #sig12=xp(sigdraw[0:2,2],axis=0)
    sig12=sigdraw[0:2,2]
    sig21=sig12.T
    sig22=sigdraw[0:2,0:2]
    cmeane3=xp((dot(dot(sig12,inv(sig22)),e12)),1)
    cvare3=sigdraw[2,2]-dot(dot(sig12,inv(sig22)),sig21)
    ytilde1=(y-cmeane3)/cvare3
    xtilde=xmat/cvare3
    bmean=dot(inv(dot(xtilde.T,xtilde)),dot(xtilde.T,ytilde1))
    bdraw=rnorm(bmean,1)
    storeb[i,:]=bdraw.T
    gamma=bdraw[2]
    ytilde2=y-xp(xmat[:,2],1)*gamma
    a=np.array([[1, 0, 0],[0, 1, 0]])
    lr=conc((bdraw[0:2,].T,ones((1,1))),1)
    a=conc((a,lr))
    omega=dot(dot(a,sigdraw),a.T)
    iu=inv(chol(omega))
    xxy=conc((conc((x1,x2),1),ytilde2),1)
    uxxy=dot(iu,xxy.T).T
    uxxy=conc((conc((uxxy[:,0],uxxy[:,1])),uxxy[:,2]))
    z1z=conc((z1,zeros((nobs,1))))
    zz2=conc((zeros((nobs,1)),z2))
    b1z1=bdraw[0]*z1
    b2z2=bdraw[1]*z2
    z1zzz2=conc((z1z,zz2),1)
    b1z1b2z2=conc((b1z1,b2z2))
    bigz=conc((z1zzz2,b1z1b2z2),1)
    ubigz=dot(iu,bigz.T).T
    temp=conc((xp(ubigz[0:nobs,2],1),xp(ubigz[nobs:2*nobs,2],1)),axis=1)
    ubigz=conc((ubigz[:,0:2],temp))
    dmean=dot(inv(dot(ubigz.T,ubigz)),dot(ubigz.T,uxxy))
    ddraw=rmvnorm(dmean,eye(2))
    #ddraw=(d1t,d2t)
    #ddraw=xp(ddraw,1)
    stored[i,:]=ddraw.T

print("mean beta", mean(storeb,0))
print("mean delta", mean(stored,0))
print("mean sigma", mean(stores,0))

# print("Sigmatrue", Sigmatrue)
# print("ldl", ldl)
# print("l", l)
# print("d", d)