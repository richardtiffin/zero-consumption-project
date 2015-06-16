__author__ = 'richard tiffin'
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
from scipy.stats import distributions
from math import *
from random import random
import statistics as stat
import scipy
import scipy.linalg
import scipy.stats as sp
from sys import exit
# TODO the code for the wishart draw comes out of github and should probably be checked
import invwishart as iw

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
plot=plt.plot
AAAA
gibbsno=5000
mcno=20
doplot=1

#### Generate data
t = 500
#True values
# Covariance
Sigmatrue = np.zeros((3,3))
Sigmatrue[0,0] = 4
Sigmatrue[0,1] = -0.5
Sigmatrue[0,2] = -0.5
Sigmatrue[1,0] = Sigmatrue[0,1]
Sigmatrue[1,1] = 2
Sigmatrue[1,2] = -0.5
Sigmatrue[2,0] = Sigmatrue[0,2]
Sigmatrue[2,1] = Sigmatrue[1,2]
Sigmatrue[2,2] = 3
d1t = 0.2
d2t = 0.2
gt=3
b1t=4
b2t=5
burn=0.1*gibbsno
plotint=1000
z1=20*runif(t,1)
z2=20*runif(t,1)
w=ones((t,1))
storebmc=zeros((mcno,3))
storedmc=zeros((mcno,2))
storesmc=zeros((mcno,9))

for j in range(0,mcno):
    #starting values
    phi=1*eye(3)
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
    bdt=bdraw
    d1draw=1
    d2draw=1
    #storage arrays
    storeb=zeros((gibbsno,3))
    stored=zeros((gibbsno,2))
    stores=zeros((gibbsno,9))

    l=chol(Sigmatrue)
    e=rnorm(0,1,(t,3))
    e=dot(l.T,e.T).T
    x1=z1*d1t+xp(e[:,0],1)
    x2=z2*d2t+xp(e[:,1],1)
    y=w*gt+x1*b1t+x2*b2t+xp(e[:,2],1)
    temp=conc((w,x1),1)
    xmat = conc((temp,x2),1)
    shapex=shape(xmat)
    nobs=shapex[0]
    z=0

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

    out=ldl(Sigmatrue)
    htrue=out[1]
    atrue=out[0]
    # print(htrue)
    # print(atrue)

    for i in range(0,gibbsno):
        print("mc loop",j,"gibbs loop",i)
        # phi=inv(phi)
        e1=x1-z1*d1draw
        e2=x2-z2*d2draw
        # bdraw=bdt
        e3=y-dot(xmat,bdraw)
        alle=conc((e1,e2,e3),1)
        phi=dot(alle.T,alle)
        # sigdraw=iw.invwishartrand(t-1,phi)
        # sigdraw=Sigmatrue

        s1=dot(e1.T,e1)
        om1=sp.invgamma.rvs(t,scale=s1)
        om1=float(om1)
        p21bar=dot(inv(s1),dot(e1.T,e2))
        p21var=om2*inv(s1)
        p21draw=rnorm(p21bar,p21var)
        u2=e2-e1*p21draw
        s2=dot(u2.T,u2)
        om2=sp.invgamma.rvs(t,scale=s2)
        om2=float(om2)
        e1e2=conc((e1,e2),1)
        is3=inv(dot(e1e2.T,e1e2))
        p31p32bar=dot(is3,dot(e1e2.T,e3))
        p31p32bar=sq(p31p32bar)
        p31p32var=om3*is3
        p31p32draw=xp(rmvnorm(p31p32bar,p31p32var),1)
        u3=e3-dot(e1e2,p31p32draw)
        s3=dot(u3.T,u3)
        om3=sp.invgamma.rvs(t,scale=s3)
        om3=float(om3)
        # om3=htrue[2,2]
        p31draw=p31p32draw[0]
        # p31draw=xp(atrue[2,0],1)
        p32draw=p31p32draw[1]
        # p21draw=0
        ia=zeros((2,3))
        ia[0,0]=1
        ia[1,0]=-1*p21draw
        ia[1,1]=1
        # ia=np.array([1,0,0],[-1*p21draw,1,0])
        row3=conc((-1*p31draw,-1*p32draw,ones(1)),1)
        row3=xp(row3,1)
        ia=conc((ia,row3.T))
        a1=inv(ia)
        # a1[:,2]=atrue[:,2]
        # a1[2,:]=atrue[2,:]
        # a1=atrue
        h=np.array([[om1,0,0],[0,om2,0],[0,0,om3]])
        # h=htrue
        sigdraw=dot(a1,dot(h,a1.T))
        stores[i,:]=np.reshape(sigdraw,(1,9))

        e12=conc((e1.T,e2.T),axis=0)
        #sig12=xp(sigdraw[0:2,2],axis=0)
        sig12=sigdraw[0:2,2]
        sig21=sig12.T
        sig22=sigdraw[0:2,0:2]
        # cmeane3a=xp((dot(dot(sig12,inv(sig22)),e12)),1)
        cmeane3=dot(e1e2,p31p32draw)
        # cvare3a=sigdraw[2,2]-dot(dot(sig12,inv(sig22)),sig21)
        cvare3=h[2,2]
        ytilde1=(y-cmeane3)/np.sqrt(cvare3)
        xtilde=xmat/np.sqrt(cvare3)
        bmean=dot(inv(dot(xtilde.T,xtilde)),dot(xtilde.T,ytilde1))
        # bmean=dot(inv(dot(xmat.T,xmat)),dot(xmat.T,y))
        varb=inv(dot(xtilde.T,xtilde))
        # varb=inv(dot(xmat.T,xmat))
        bmeansq=sq(bmean,axis=1)
        # varb=eye(3)
        bdraw=rmvnorm(bmeansq,varb)
        # bdraw=bdt
        bdraw=xp(bdraw,1)
        storeb[i,:]=bdraw.T
        gamma=bdraw[0]
        ytilde2=y-xp(xmat[:,0],1)*gamma
        a=np.array([[1, 0, 0],[0, 1, 0]])
        lr=conc((bdraw[1:3,].T,ones((1,1))),1)
        a=conc((a,lr))
        omega=dot(dot(a,sigdraw),a.T)
        iu=inv(chol(omega))
        xxy=conc((conc((x1,x2),1),ytilde2),1)
        uxxy=dot(iu.T,xxy.T).T
        uxxy=conc((conc((uxxy[:,0],uxxy[:,1])),uxxy[:,2]))
        z1z=conc((z1,zeros((nobs,1))))
        zz2=conc((zeros((nobs,1)),z2))
        b1z1=bdraw[1]*z1
        b2z2=bdraw[2]*z2
        z1zzz2=conc((z1z,zz2),1)
        b1z1b2z2=conc((b1z1,b2z2))
        bigz=conc((z1zzz2,b1z1b2z2),1)
        ubigz=dot(iu.T,bigz.T).T
        temp=conc((xp(ubigz[0:nobs,2],1),xp(ubigz[nobs:2*nobs,2],1)),axis=1)
        ubigz=conc((ubigz[:,0:2],temp))
        dvar=inv(dot(ubigz.T,ubigz))
        dmean=dot(dvar,dot(ubigz.T,uxxy))
        ddraw=rmvnorm(dmean,dvar)
        # ddraw=(d1t,d2t)
        # ddraw=xp(ddraw,1)
        d1draw=ddraw[0]
        d2draw=ddraw[1]
        stored[i,:]=ddraw.T
        z=z+1
        if doplot==1:
            if z==plotint:
                plot(stores[10:gibbsno,:])
                plt.pause(0.001)
                z=0
    storebmc[j,:]=mean(storeb[burn:gibbsno,:],0)
    storedmc[j,:]=mean(stored[burn:gibbsno,:],0)
    storesmc[j,:]=mean(stores[burn:gibbsno,:],0)


print("mean beta", mean(storebmc,0))
print("mean delta", mean(storedmc,0))
print("mean sigma", mean(storesmc,0))
# plot(storeb)
# plt.show()
# plot(stores)
# plt.show()

# print("Sigmatrue", Sigmatrue)
# print("ldl", ldl)
# print("l", l)
# print("d", d)