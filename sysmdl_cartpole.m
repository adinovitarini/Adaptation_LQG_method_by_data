function cartpole = sysmdl_cartpole(N)
close all;clc
g = 9.8e-1;
lp = 1.5;
mp = 0.1;
mk = 1;
mt = 1.1;
a = g/(lp*(4/3 -mp/(mp+mk)));
AA = [0 1 0 0;0 0 mp/mk*g 0;0 0 0 1;0 0 (mt)*g/(lp*mk) 0];
b = -1/(lp*(4/3 -mp/(mp+mk)));
BB = [0;1/mk;0;1/(lp*mk)];
CC = [0 1 0 1];
D = 0;
N = 100;
df = 0.1; %discount factor 
A = sqrt(df)*AA;
B = sqrt(df)*BB;
C = sqrt(df)*CC;
sys = ss(A,B,C,D,0.01);
%%  Apply Disturbance 
w = wgn(N,100,1);
v = w;
cartpole.sys = sys;