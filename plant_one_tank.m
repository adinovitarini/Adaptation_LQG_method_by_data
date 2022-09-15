% clearvars -except data 
tic
clear all;clc
z = tf('z');
Gz = tf([0 0.08123],[1 -0.9895],0.01);
[A,B,C,D] = tf2ss([0 0.08123],[1 -0.9895]);
sys = ss(A,B,C,D,0.01);
N = 100;
u = 0;
%%  Apply Disturbance 
w = wgn(N,1,1);
v = w;
Rww = 1;
Rvv = 1;
% w = unifrnd(0,Rww,N,1);
% v = unifrnd(0,Rvv,N,1);
%%  Open Loop System
x = .1*ones(size(A,1),1);
for i = 1:N
    x(:,i+1) = A*x(:,i)+B*u+w(i);
    y(i) = C*x(:,i)+v(i);
    x(:,i) = x(:,i+1);
end
%%  LQG Conventional 
R = Rvv;
for i = 1:5
    Rww(i) = 10^-i;
    [~,LL(:,i),~] = kalman(sys,Rww(i),R);
    [L(:,:,i),~] = KalmanConventional(sys,N,Rww(i),Rvv);
    [K(i,:),~,cl(:,i)] = lqr(sys,Rww(i),R);
end
for i = 1:5
    for j = 1:size(A,1)
        L_norm(j,i) = norm(L(j,:,i));
    end
    L_normm(i) = norm(L_norm(:,i));
end
[val,idx] = min(L_normm);
L_kf = L(:,:,idx);
LL_kf = LL(:,idx);
K_lqr = K(idx,:);
%% Observer-based-controller 
x_hat = ones(size(A,1),1,1);
for i = 1:N
    for j = 1:1
        u_c(j,i) = -K_lqr(j,:)*x_hat(:,i,j);
        y_hat(j,i) = C*x_hat(:,i,j);
        x_hat(:,i+1,j) = (A-B*K_lqr(j,:))*x_hat(:,i,j)+L(:,j)*(y_hat(i)-y(i));
    end
end
%%  Observer via KalmanNet
x_nw = ones(size(A,1),1);
for i = 1:N
    x_nw(:,i+1) = A*x_nw(:,i);
    y_nw(i) = C*x_nw(:,i);
    x_nw(:,i) = x_nw(:,i+1);
end
x_nw = x_nw(:,1:N);
delta_x = [x(:,2:end);y];
target = [x_nw;y_nw];
KG = KalmanNet(delta_x,target,A,C);
%%  Implement KG for state estimation process 
delta_y = y+(-1*y_nw);
x_hat_net = (KG*delta_y')+x_nw;
y_hat_net = C*x_hat_net;
%%  First scenario : KF-VI   
Q = 1;
R = .1;
[P_kf_vi,u_kf_vi,x_hat_kf_vi,y_kf_vi,J_kf_vi,K_kf_vi,L_kf_vi] = combine_kf_vi(A,B,C,Q,R,N,L_kf,y);
% for i = 1:size(x_hat_kf_vi,1)
%     x_hat_kf_vi(i,:) = normalize(x_hat_kf_vi(i,:));
% end
%%  Second scenario : KalmanNet-LQR   
% Implement KalmanNet-LQR 
[P_kn_lqr,u_kn_lqr,x_hat_kn_lqr,y_kn_lqr,J_kn_lqr,K_kn_lqr] = combine_kn_lqr(A,B,C,Q,R,N,x_hat_net);
% for i = 1:size(x_hat_kn_lqr,1)
%     x_hat_kn_lqr(i,:) = normalize(x_hat_kn_lqr(i,:));
% end
%%  Third scenario : KalmanNet-VI
%Implement Value Iteration 
[P_kn_vi,u_kn_vi,x_hat_kn_vi,y_kn_vi,J_kn_vi,K_kn_vi] = combine_kn_vi(A,B,C,Q,R,N,x_hat_net);
% for i = 1:size(x_hat_kn_vi,1)
%     x_hat_kn_vi(i,:) = normalize(x_hat_kn_vi(i,:));
% end
%% Stability Analysis 
for i = 1:N
eig_kf_vi(:,i) = eig(A-B*K_kf_vi(i,:)-L_kf_vi(:,i)*C);
eig_kn_lqr(:,i) = eig(A-B*K_kn_lqr-KG(1,i)*C);
eig_kn_vi(:,i) = eig(A-B*K_kn_vi(i,:)-KG(1,i)*C);
end
norm_eig_kf_vi = zeros(size(A,1),1);
norm_eig_kn_lqr = norm_eig_kf_vi;
norm_eig_kn_vi = norm_eig_kn_lqr;
for j = 1:size(A,1)
    norm_eig_kf_vi(j,1) = max(eig_kf_vi(j,:));
    norm_eig_kn_lqr(j,1) = max(eig_kn_lqr(j,:));
    norm_eig_kn_vi(j,1) = max(eig_kn_vi(j,:));
end
time_elapsed = toc
%% Display to command window 
clc
fprintf('Norm Ac 1^{st} scenario : \n %f \n',norm_eig_kf_vi)
fprintf('Norm Ac 2^{nd} scenario : \n %f \n',norm_eig_kn_lqr)
fprintf('Norm Ac 3^{rd} scenario : \n %f \n',norm_eig_kn_vi)
%%  Visualize the eigenvalues
figure(1);clf
plot(real(eig_kf_vi(1,:)),'-ob','markersize',2)
hold on
grid on
% xlim([1 N])
xlabel('Iteration Step')
ylabel('$eig(A-BK-LC)$','Interpreter','latex')
legend('$\hat{x}_1$','$\hat{x}_2$','$\hat{x}_3$','$\hat{x}_4$','Interpreter','latex')
title('The Evolution of Eigenvalues on 1^{st} scenario : Combination KF and VI')
figure(2);clf
plot(real(eig_kn_lqr(1,:)),'-ob','markersize',2)
hold on
grid on
% xlim([1 N])
xlabel('Iteration Step')
ylabel('$eig(A-BK-LC)$','Interpreter','latex')
legend('$\hat{x}_1$','$\hat{x}_2$','$\hat{x}_3$','$\hat{x}_4$','Interpreter','latex')
title('The Evolution of Eigenvalues on 2^{nd} scenario : Combination KN and LQR')
figure(3);clf
plot(real(eig_kn_vi(1,:)),'-ob','markersize',2)
grid on
% xlim([1 N])
xlabel('Iteration Step')
ylabel('$eig(A-BK-LC)$','Interpreter','latex')
legend('$\hat{x}_1$','$\hat{x}_2$','$\hat{x}_3$','$\hat{x}_4$','Interpreter','latex')
title('The Evolution of Eigenvalues on 3^{rd} scenario : Combination KN and VI')
figure(4);clf
plot(u_kf_vi','-ob','markersize',2);
hold on
plot(u_kn_lqr','-or','markersize',2);
hold on
plot(u_kn_vi','-ok','markersize',2);
hold on
grid on
title('Control Signal')
xlim([1 N])
xlabel('Iteration step')
ylabel('$u_k(x_k)$','Interpreter','latex')
legend('1^{st} scenario : Combination KF-VI','2^{nd} scenario : Combination KN-LQR','3^{rd} scenario : Combination KN-VI')

axes('Position',[0.35,0.35,0.35,0.35]);
box on
plot(u_kf_vi(:,1:4)','-ob','markersize',2);
hold on
plot(u_kn_lqr(:,1:4)','-or','markersize',2);
hold on
plot(u_kn_vi(:,1:4)','-ok','markersize',2);
% set(gca,'xtick',[])
set(gca,'ytick',[])
figure(5);clf 
title('The state trajectories')
plot(x_hat_kf_vi(:,1:N),'-ob','markersize',2);
hold on
plot(x_hat_kn_lqr(:,1:N),'-or','markersize',2);
hold on
plot(x_hat_kn_vi(:,1:N),'-ok','markersize',2);
hold on
grid on
xlabel('Iteration step')
ylabel('$x_k$','Interpreter','latex')
legend('1^{st} scenario : Combination KF-VI','2^{nd} scenario : Combination KN-LQR','3^{rd} scenario : Combination KN-VI')

axes('Position',[0.35,0.35,0.35,0.35]);
box on
plot(x_hat_kf_vi(:,1:4),'-ob','markersize',2);
hold on
plot(x_hat_kn_lqr(:,1:4),'-or','markersize',2);
hold on
plot(x_hat_kn_vi(:,1:4),'-ok','markersize',2);
hold on
% set(gca,'xtick',[])
set(gca,'ytick',[])