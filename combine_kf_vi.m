%% First Scenario : Kalman Filter and Value Iteration 
% Q : 5 candidates of matrix Q, please save into row matrix 
%for example Q = [0.01;0.05;0.1;0.5;1];
function [P,u,x_hat,y,J,K,L] = combine_kf_vi(A,B,C,Q,R,N,L_kf,y)
for i = 1:5
    [P_vi,K_vi_old(:,:,i),G_vi(i,:)] = value_iteration(A,N,B,Q(i),R);
    mans(i) = norm(K_vi_old(:,:,i));
end
[~,idx] = max(mans);
K_vi = K_vi_old(:,:,idx);
x_hat_kf_vi = ones(4,1);
for i = 1:N
    u_kf_vi(i) = -K_vi(i,:)*x_hat_kf_vi(:,i);
    y_hat_kf_vi(i) = C*x_hat_kf_vi(:,i);
    x_hat_kf_vi(:,i+1) = (A-B*K_vi(i,:))*x_hat_kf_vi(:,i)+L_kf(:,i)*(y_hat_kf_vi(i)-y(i));         
end
% Value function 
for i = 1:5
    J_kf_vi(i,:) = value_func(x_hat_kf_vi,u_kf_vi,Q(i),R,N);
end
% output 
P = P_vi;
u = u_kf_vi;
y = y_hat_kf_vi;
x_hat = x_hat_kf_vi;
J = J_kf_vi;
K = K_vi;
L = L_kf;