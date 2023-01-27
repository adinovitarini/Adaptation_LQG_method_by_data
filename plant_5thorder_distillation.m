% clearvars -except data 
tic
clear all;clc
z = tf('z');
AA = [1.1367 -0.77978 -0.41183 -0.93463 0;
    1.0468 1.0221 0.51514 0.55115 0;
    -0.77322 0.73872 -0.83021 0.0026816 0;
    1.1816 0.95094 -0.65203 -0.78764 0;
    0 0 0 0 1];
BB = [-1.3696;0.45253;1.0801;-0.37804;1];
CC = [0.72552 -0.78382 0.97289 -0.3413 1];
D = 0;
df = .01; %discount factor 
A = sqrt(df)*AA;
B = sqrt(df)*BB;
C = sqrt(df)*CC;
sys = ss(A,B,C,D,0.01);
N = 100;
u = 0;
%%  Apply Disturbance 
w = wgn(N,1,1);
v = w;
%%  Open Loop System
x = .1*ones(size(A,1),1);
for i = 1:N
    x(:,i+1) = A*x(:,i)+B*u+w(i);
    y(i) = C*x(:,i)+v(i);
    x(:,i) = x(:,i+1);
end
%% Weight Matrices 
Rww = 0.01;
Rvv = .1;
Q = .01;
R = .1;
%%  LQG Conventional 
[~,L,~] = kalman(sys,Rww,Rvv);
[K,~,cl] = lqr(sys,Q,R);
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
[KG,net] = KalmanNet(delta_x,target,A,C,N);
% [KG,x_hat_net,time_elapsed,error_rate] = test_kn(3,0.00001,delta_x,target,A,C,y,y_nw,x_nw)
%%  Implement KG for state estimation process 
delta_y = y+(-1*y_nw);
x_hat_net = (KG*delta_y')+x_nw;
y_hat_net = C*x_hat_net;
%%  First scenario : KF-VI   
[kf_vi] = combine_kf_vi(A,B,C,Q,R,N,L,y);
P_kf_vi = kf_vi.P;
u_kf_vi = kf_vi.u; 
x_hat_kf_vi = kf_vi.x_hat; 
y_kf_vi = kf_vi.y;
J_kf_vi = kf_vi.J;
K_kf_vi = kf_vi.K;
L_kf_vi = kf_vi.L;
%%  Second scenario : KalmanNet-LQR   
[kn_lqr] = combine_kn_lqr(A,B,C,Q,R,N,x_hat_net);
P_kn_lqr = kn_lqr.P;
u_kn_lqr = kn_lqr.u; 
x_hat_kn_lqr = kn_lqr.x_hat; 
y_kn_lqr = kn_lqr.y;
J_kn_lqr = kn_lqr.J;
K_kn_lqr = kn_lqr.K;
%%  Third scenario : KalmanNet-VI
[kn_vi] = combine_kn_vi(A,B,C,Q,R,N,x_hat_net);
P_kn_vi = kn_vi.P;
u_kn_vi = kn_vi.u; 
x_hat_kn_vi = kn_vi.x_hat; 
y_kn_vi = kn_vi.y;
J_kn_vi = kn_vi.J;
K_kn_vi = kn_vi.K;
% [P_kn_vi,u_kn_vi,x_hat_kn_vi,y_kn_vi,J_kn_vi,K_kn_vi] = combine_kn_vi(A,B,C,Q,R,N,x_hat_net);
%% Fourth scenario : LQG konvensional 
x_lqg = ones(size(A,1),1);
x_hat_lqg = x;
y_lqg = y;
L_lqg = L;
K_lqg = K;
for i = 1:N
    u_lqg(i) = -K_lqg*x_lqg(:,i);
    x_hat_lqg(:,i+1) = A*x_hat_lqg(:,i)+B*u_lqg(i);
    y_hat_lqg(i) = C*x_hat_lqg(:,i);
    x_lqg(:,i+1) = (A-B*K_lqg)*x_lqg(:,i)+L_lqg*(y_hat_lqg(i)-y_lqg(i));
    y_lqg(i) = C*x_lqg(:,i);
end
J_lqg = value_func(x_hat_lqg,u_lqg,Q,R,N);

% %% State Transition 
% Acl_kf_vi = [A-B*K_kf_vi(end,:) B*K_kf_vi(end,:);zeros(size(A,1),size(A,1)) A-L_kf_vi*C].^100;
% Acl_kn_lqr = [A-B*K_kn_lqr B*K_kn_lqr;zeros(size(A,1),size(A,1)) A-KG(1,end)*C].^100;
% Acl_kn_vi = [A-B*K_kn_vi(end,:) B*K_kn_vi(end,:);zeros(size(A,1),size(A,1)) A-KG(1,end)*C].^100;
% Acl_lqg = [A-B*K_lqg B*K_lqg;zeros(size(A,1),size(A,1)) A-L_lqg*C].^100;
% %% Stability Analysis 
% for i = 1:N
% eig_kf_vi(:,i) = eig([A-B*K_kf_vi(i,:) B*K_kf_vi(i,:);zeros(size(A,1),size(A,1)) A-L_kf_vi*C]);
% eig_kn_lqr(:,i) = eig([A-B*K_kn_lqr B*K_kn_lqr;zeros(size(A,1),size(A,1)) A-KG(1,i)*C]);
% eig_kn_vi(:,i) = eig([A-B*K_kn_vi(i,:) B*K_kn_vi(i,:);zeros(size(A,1),size(A,1)) A-KG(1,i)*C]);
% eig_lqg(:,i) = eig([A-B*K_lqg B*K_lqg;zeros(size(A,1),size(A,1)) A-L_lqg*C]);
% end
% norm_eig_kf_vi = zeros(size(A,1),1);
% norm_eig_kn_lqr = norm_eig_kf_vi;
% norm_eig_kn_vi = norm_eig_kn_lqr;
% for j = 1:size(eig_kf_vi,1)
%     [norm_eig_kf_vi(j,1),idx1(j,1)] = max(eig_kf_vi(j,:));
%     [norm_eig_kn_lqr(j,1),idx2(j,1)] = max(eig_kn_lqr(j,:));
%     [norm_eig_kn_vi(j,1),idx3(j,1)] = max(eig_kn_vi(j,:));
%     [norm_eig_lqg(j,1),idx4(j,1)] = max(eig_lqg(j,:));
% end
clc
% fprintf('CL-Eigenvalue scenario 1 : \n %f \n%f \n%f \n%f \n%f \n%f \n%f \n%f \n',eig_kf_vi(:,1))
% fprintf('CL-Eigenvalue scenario 2 : \n %f \n%f \n%f \n%f \n%f \n%f \n%f \n%f \n',eig_kn_lqr(:,1))
% fprintf('CL-Eigenvalue scenario 3 : \n %f \n%f \n%f \n%f \n%f \n%f \n%f \n%f \n',eig_kn_vi(:,1))
% fprintf('CL-Eigenvalue scenario 4 : \n %f \n%f \n%f \n%f \n%f \n%f \n%f \n%f \n',eig_lqg(:,1))

%% Plot
% figure(1);clf
% plot(real(eig_kf_vi(1,:)),'-ob','markersize',2)
% hold on
% plot(real(eig_kf_vi(2,:)),'-om','markersize',2)
% hold on
% plot(real(eig_kf_vi(3,:)),'-or','markersize',2)
% hold on
% plot(real(eig_kf_vi(4,:)),'-ok','markersize',2)
% hold on
% plot(real(eig_kf_vi(5,:)),'*b','markersize',2)
% hold on
% plot(real(eig_kf_vi(6,:)),'*m','markersize',2)
% hold on
% plot(real(eig_kf_vi(7,:)),'*r','markersize',2)
% hold on
% plot(real(eig_kf_vi(8,:)),'*k','markersize',2)
% hold on
% grid on
% % xlim([1 N])
% xlabel('Iteration Step')
% ylabel('$eig(A_c(k))$','Interpreter','latex')
% legend('$x_1$','$x_2$','$x_3$','$x_4$','$\tilde{x}_1$','$\tilde{x}_2$','$\tilde{x}_3$','$\tilde{x}_4$','Interpreter','latex')
% title('The Evolution of Eigenvalues on 1^{st} scenario : Combination KF and VI')
% figure(2);clf
% plot(real(eig_kn_lqr(1,:)),'-ob','markersize',2)
% hold on
% plot(real(eig_kn_lqr(2,:)),'-om','markersize',2)
% hold on
% plot(real(eig_kn_lqr(3,:)),'-or','markersize',2)
% hold on
% plot(real(eig_kn_lqr(4,:)),'-ok','markersize',2)
% hold on
% plot(real(eig_kn_lqr(5,:)),'*b','markersize',2)
% hold on
% plot(real(eig_kn_lqr(6,:)),'*m','markersize',2)
% hold on
% plot(real(eig_kn_lqr(7,:)),'*r','markersize',2)
% hold on
% plot(real(eig_kn_lqr(8,:)),'*k','markersize',2)
% hold on
% grid on
% % xlim([1 N])
% xlabel('Iteration Step')
% ylabel('$eig(A_c(k))$','Interpreter','latex')
% legend('$x_1$','$x_2$','$x_3$','$x_4$','$\tilde{x}_1$','$\tilde{x}_2$','$\tilde{x}_3$','$\tilde{x}_4$','Interpreter','latex')
% title('The Evolution of Eigenvalues on 2^{nd} scenario : Combination KN and LQR')
% figure(3);clf
% plot(real(eig_kn_vi(1,:)),'-ob','markersize',2)
% hold on
% plot(real(eig_kn_vi(2,:)),'-om','markersize',2)
% hold on
% plot(real(eig_kn_vi(3,:)),'-or','markersize',2)
% hold on
% plot(real(eig_kn_vi(4,:)),'-ok','markersize',2)
% hold on
% plot(real(eig_kn_vi(5,:)),'*b','markersize',2)
% hold on
% plot(real(eig_kn_vi(6,:)),'*m','markersize',2)
% hold on
% plot(real(eig_kn_vi(7,:)),'*r','markersize',2)
% hold on
% plot(real(eig_kn_vi(8,:)),'*k','markersize',2)
% hold on
% grid on
% % % xlim([1 N])
% xlabel('Iteration Step')
% ylabel('$eig(A_c(k))$','Interpreter','latex')
% legend('$x_1$','$x_2$','$x_3$','$x_4$','$\tilde{x}_1$','$\tilde{x}_2$','$\tilde{x}_3$','$\tilde{x}_4$','Interpreter','latex')
% title('The Evolution of Eigenvalues on 3^{rd} scenario : Combination KN and VI')
% figure(4);clf
% plot(real(eig_lqg(1,:)),'-ob','markersize',2)
% hold on
% plot(real(eig_lqg(2,:)),'-om','markersize',2)
% hold on
% plot(real(eig_lqg(3,:)),'-or','markersize',2)
% hold on
% plot(real(eig_lqg(4,:)),'-ok','markersize',2)
% hold on
% plot(real(eig_lqg(5,:)),'*b','markersize',2)
% hold on
% plot(real(eig_lqg(6,:)),'*m','markersize',2)
% hold on
% plot(real(eig_lqg(7,:)),'*r','markersize',2)
% hold on
% plot(real(eig_lqg(8,:)),'*k','markersize',2)
% hold on
% grid on
% % % xlim([1 N])
% xlabel('Iteration Step')
% ylabel('$eig(A_c(k))$','Interpreter','latex')
% legend('$x_1$','$x_2$','$x_3$','$x_4$','$\tilde{x}_1$','$\tilde{x}_2$','$\tilde{x}_3$','$\tilde{x}_4$','Interpreter','latex')
% title('The Evolution of Eigenvalues on 4^{th} scenario : LQG')
%% Display the output 
% clc
% fprintf('Kalman gain from KF : %f \n',norm(L))
% fprintf('Kalman gain from KN : %f \n',norm(KG))
% fprintf('Controller gain from LQR : %f \n',norm(K))
% fprintf('Controller gain from VI : %f \n',norm(K_kf_vi))
% fprintf('Controller gain from LQR : %f \n',norm(K_kn_lqr))
% fprintf('Controller gain from VI : %f \n',norm(K_kn_vi))
% fprintf('Error norm from 1st scenario : %f \n',norm(x-x_lqg));
% fprintf('Error norm from 2nd scenario : %f \n',norm(x-x_hat_kf_vi));
% fprintf('Error norm from 3rd scenario : %f \n',norm(x-x_hat_kn_lqr));
% fprintf('Error norm from 4th scenario : %f \n',norm(x-x_hat_kn_vi));