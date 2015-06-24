__author__ = 'Ariane Kehlbacher'
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import distributions
from math import *
from random import random
import statistics as stat
import scipy
import scipy.linalg
import scipy.stats as sp
rnorm = np.random.normal
rmvnorm=np.random.multivariate_normal
runif = np.random.rand
zeros = np.zeros
ones = np.ones
dot = np.dot
add = np.add
mean = np.mean
eye = np.identity
transpose = np.transpose
diag = np.diag
shape = np.shape
chol = np.linalg.cholesky
inv = np.linalg.inv
xp = np.expand_dims
sq = np.squeeze


#************ Generate data ***********#
draws = np.matrix([10000])
burn = np.matrix([0.2])*draws
draws += burn
t = 1000
s = 3  # number of coefficients in w (exogenous)
p1 = 3 # number of coefficients in z1 (instruments)
p2 = 3 # number of coefficients in z2 (instruments)

a31t = 0.8 # correlation between e1 and e3
a32t = 0.4 # correlation between e2 and e3

A = np.zeros((3,3))
A[0,0] = 1
A[1,0] = 0
A[1,1] = 1
A[2,0] = a31t
A[2,1] = a32t
A[2,2] = 1

H = np.zeros((3,3))
sig11t = 2
sig22t = 3
sig3_12t = 2.5

H[0,0] = sig11t
H[1,1] = sig22t
H[2,2] = sig3_12t

Sigmatrue = A.dot(H).dot(A.T)

z1=20*runif(t,p1)
z2=20*runif(t,p2)
d1t = xp(rnorm(8,1,p1),1)
d2t = xp(rnorm(5,1,p2),1)
w = np.c_[np.ones((t)),rnorm(6,1,(t,s-1))]
gt = xp(rnorm(6,1,s),1)
l = chol(Sigmatrue)
e0 = rnorm(0,1,(t,3))
e = e0.dot(l.T)
b1t = rnorm(3,1,1)
b2t = rnorm(4,1,1)
x1 = z1.dot(d1t)+xp(e[:,0],axis=1)
x2 = z2.dot(d2t)+xp(e[:,1],axis=1)
y = w.dot(gt)+x1*b1t+x2*b2t+xp(e[:,2],axis=1)


# Starting values #
a31draw = 1
a32draw = 1
Sigma = eye(3)
sig11draw = ones((1,1))
sig22draw = ones((1,1))
sig3_12draw = ones((1,1))
d1draw = ones((p1,1))
d2draw = ones((p1,1))
# Storage matrices
storegb = zeros((draws,s+2))
stored1 = zeros((draws,p1))
stored2 = zeros((draws,p2))
storesig = zeros((draws,6))
storea = zeros((draws,2))


for i in range(0,draws):
    print("gibbs loop",i)
    #d1draw = d1t
    #d2draw = d2t
    #sig3_12draw = sig3_12t
    #sig11draw = np.array([sig11t])[None,:]
    #sig22draw = np.array([sig22t])[None,:]
    #b1draw = b1t
    #b2draw = b2t
    #a31draw = a31t
    #a32draw = a32t


    # Draw gamma, beta1 and beta2
    a13 = Sigma[2,0]*inv(sig11draw)
    a23 = Sigma[2,1]*inv(sig22draw)
    e1 = x1-z1.dot(d1draw)
    e2 = x2-z2.dot(d2draw)
    yt = np.divide((y-(a13*e1)-(a23*e2)),sqrt(sig3_12draw))
    xt = np.divide(np.c_[w,x1,x2],sqrt(sig3_12draw))
    gbmean = dot(inv(dot(xt.T,xt)),dot(xt.T,yt))
    gbcov = inv(dot(xt.T,xt))
    gbdraw = rmvnorm(sq(gbmean,axis=1),gbcov)
    storegb[i,:] = gbdraw
    gdraw = xp(gbdraw[0:s],axis=1)
    b1draw = xp(xp(gbdraw[s],axis=1),axis=1)
    b2draw = xp(xp(gbdraw[s+1],axis=1),axis=1)
    e3 = y-w.dot(gdraw)-z1.dot(b1draw*d1draw)-z2.dot(b2draw*d2draw)-b1draw*e1-b2draw*e2

    # Draw delta1 and delta2
    sig1_3 = Sigmatrue[0,0]-np.divide(Sigmatrue[0,2]**2,Sigmatrue[2,2])
    sig2_3 = Sigmatrue[1,1]-np.divide(Sigmatrue[1,2]**2,Sigmatrue[2,2])
    v3 = y-w.dot(gdraw)-b1draw*x1-b2draw*x2
    xtd1 = np.divide(x1-v3*a31draw*a32draw,sqrt(sig1_3))
    xtd2 = np.divide(x2-v3*a31draw*a32draw,sqrt(sig2_3))
    ztd1 = np.divide(z1,sqrt(sig1_3))
    ztd2 = np.divide(z2,sqrt(sig2_3))
    d1mean = dot(inv(dot(ztd1.T,ztd1)),dot(ztd1.T,xtd1))
    d2mean = dot(inv(dot(ztd2.T,ztd2)),dot(ztd2.T,xtd2))
    d1cov = inv(dot(ztd1.T,ztd1))
    d2cov = inv(dot(ztd1.T,ztd1))
    d1draw=rmvnorm(sq(d1mean,axis=1),d1cov)
    d2draw=rmvnorm(sq(d2mean,axis=1),d2cov)
    stored1[i,:] = d1draw
    stored2[i,:] = d2draw
    d1draw = xp(d1draw,axis=1)
    d2draw = xp(d2draw,axis=1)


    # Draw a31 and a32
    e12 = np.c_[e1,e2]
    amean = dot(inv(dot(e12.T,e12)),dot(e12.T,e3))
    acov = inv(dot(e12.T,e12))
    adraw = rmvnorm(sq(amean,axis=1),acov)
    storea[i,:] = adraw
    a31draw = adraw[0]
    a32draw = adraw[1]

    # Draw sigma11, sigma22
    sig11draw = np.array([scipy.stats.gamma.rvs(e1.T.dot(e1), loc=0, scale = 1.0/t)])
    sig11draw = sig11draw[None, :]
    sig22draw = np.array([scipy.stats.gamma.rvs(e2.T.dot(e2), loc=0, scale = 1.0/t)])
    sig22draw = sig22draw[None, :]
    # Draw sigma 3|12
    v3 = e3 - a31t*e1 - a32t*e2
    sig3_12draw = np.array([scipy.stats.gamma.rvs(v3.T.dot(v3), loc=0, scale = 1.0/t)])
    sig3_12draw = sig3_12draw[None, :]

    Sigma[0,0] = sig11draw
    Sigma[1,1] = sig22draw
    Sigma[0,2] = a31draw*sig11draw
    Sigma[2,0] = Sigma[0,2]
    Sigma[1,2] = a32draw*sig22draw
    Sigma[2,1] = Sigma[1,2]
    Sigma[2,2] = sig3_12draw+np.divide(Sigma[0,2]**2,sig11draw)+np.divide(Sigma[1,2]**2,sig22draw)
    storesig[i,:] = Sigma[np.triu_indices(3)]





# Printing
print("")
print("delta1 true", d1t.T)
print("delta1 est", mean(stored1[burn+1:draws,:],0))
print("")
print("delta2 true", d2t.T)
print("delta2 est", mean(stored2[burn+1:draws,:],0))
print("")
print("gamma~b1~b2 true", gt.T, b1t,b2t)
print("gamma~b1_b2 est", mean(storegb[burn+1:draws,:],0))
print("")
print("a31~a32 true", a31t, a32t)
print("a31_a32 est", mean(storea[burn+1:draws,:],0))
print("")
print("Sigma true", Sigmatrue[np.triu_indices(3)])
print("Sigma est", mean(storesig[burn+1:draws,:],0))


