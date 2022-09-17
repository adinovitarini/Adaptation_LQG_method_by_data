function [LL,S] = KalmanConventional(sys,N,Rww,Rvv)
%% Initialisasi
A = sys.A;
B = sys.B;
C = sys.C;
D = sys.D;
x_hat_kf = 0*ones(size(A,1),1);
u = 0;
%%  Apply Disturbance 
w = wgn(N,1,1);
v = w;
% Rww = 10;
% Rvv = 1;
% w = unifrnd(0,Rww,N,1);
% v = unifrnd(0,Rvv,N,1);
%%  Init
% Rww = 0.1*ones(size(A,1),size(A,1)); %Q=Rww
% Rvv = 0.0001; %R=Rvv
P = rand(size(A,1),size(A,1),N); %Matriks covariance state estimate
%% Iteration
for i = 1:N
    x_new(:,i) = A*x_hat_kf(:,i) + B*u + w(i);       
    y(i) = C*x_hat_kf(:,i)+v(i);
    x_hat_kf(:,i+1) = x_new(:,i);
    %Prior Error Covariance
    P(:,:,i+1) = A*P(:,:,i)*A'+Rww;
    P1_new(:,:,i) = A*P(:,:,i)+P(:,:,i)*A'-P(:,:,i)*C'*inv(Rvv)*C*P(:,:,i)+Rww;
    %Measurement Update
    S(:,:,i) = C*P1_new(:,:,i)*C'+Rvv;
    %Calculate Kalman Gain 
    LL(:,i) = P(:,:,i)*C'*inv(S(:,:,i));
    x_hat_new__kf(:,i) = x_hat_kf(:,i)+LL(:,i)*(y(i)-C*x_hat_kf(:,i));
    x_hat_kf(:,i+1) = x_hat_new__kf(:,i);
end
