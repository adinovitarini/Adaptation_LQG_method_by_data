%% Second Scenario : KalmanNet and LQR
% Q : 5 candidates of matrix Q, please save into row matrix 
%for example Q = [0.01;0.05;0.1;0.5;1];
function [P,u,x_hat,y,J,K] = combine_kn_lqr(A,B,C,Q,R,N,x_hat_net,y_hat_net)
for i = 1:5
    [P(:,:,i),KK(i,:),eigg(i,:)] = idare(A,B,Q(i),R);
    P_norm(i) = norm(P(:,:,i));
end
x_hat_kn_lqr = x_hat_net;
[~,idx] = max(P_norm);
KK = KK(idx,:);
for i = 1:N
    u_kn_lqr(1,i) = -KK*x_hat_kn_lqr(:,i);
    y_hat_kn_lqr(i) = C*x_hat_kn_lqr(:,i);
    x_hat_kn_lqr(:,i+1) = A*x_hat_kn_lqr(:,i)+B*u_kn_lqr(:,i);
end
% Value function 
for i = 1:5
    J_kn_lqr(i,:) = value_func(x_hat_kn_lqr,u_kn_lqr,Q(i),R,N);
end
% % output 
u = abs(u_kn_lqr);
y = y_hat_kn_lqr;
x_hat = abs(x_hat_kn_lqr);
J = J_kn_lqr;
K = KK;