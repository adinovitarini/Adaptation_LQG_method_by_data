%% Third Scenario : KalmanNet and VI
% Q : 5 candidates of matrix Q, please save into row matrix 
%for example Q = [0.01;0.05;0.1;0.5;1];
function [P,u,x_hat,y,J,K] = combine_kn_vi(A,B,C,Q,R,N,x_hat_net,y_hat_net)
[P_vi,K_vi,G_vi] = value_iteration(A,N,B,Q(1),R);
for i = 1:5
    [P_vi,K_vi_old(:,:,i),G_vi(i,:)] = value_iteration(A,N,B,Q(i),R);
    mans(i) = norm(K_vi_old(:,:,i));
end
[~,idx] = max(mans);
x_hat_kn_vi = x_hat_net;
KK = K_vi(idx,:);
for i = 1:N
    u_kn_vi(1,i) = -KK*x_hat_kn_vi(:,i);
    y_hat_kn_vi(i) = C*x_hat_kn_vi(:,i);
    x_hat_kn_vi(:,i+1) = A*x_hat_kn_vi(:,i)+B*u_kn_vi(:,i);
end
% Value function 
for i = 1:5
    J_kn_vi(i,:) = value_func(x_hat_kn_vi,u_kn_vi,Q(i),R,N);
end
% % output
P = P_vi;
u = abs(u_kn_vi);
y = y_hat_kn_vi;
x_hat = abs(x_hat_kn_vi);
J = J_kn_vi;
K = K_vi;