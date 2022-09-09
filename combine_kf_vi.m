%% First Scenario : Kalman Filter and Value Iteration 
function [P,u,x_hat,y,J,K,L] = combine_kf_vi(A,B,C,Q,R,N,L_kf,y)
[P_vi,K_vi_old,~] = value_iteration(A,N,B,Q,R);
K_vi = K_vi_old;
x_hat_kf_vi = ones(4,1);
for i = 1:N
    u_kf_vi(i) = -K_vi(i,:)*x_hat_kf_vi(:,i);
    y_hat_kf_vi(i) = C*x_hat_kf_vi(:,i);
    x_hat_kf_vi(:,i+1) = (A-B*K_vi(i,:))*x_hat_kf_vi(:,i)+L_kf(:,i)*(y_hat_kf_vi(i)-y(i));         
end
% Value function 
J_kf_vi = value_func(x_hat_kf_vi,u_kf_vi,Q,R,N);
% output 
P = P_vi;
u = u_kf_vi;
y = y_hat_kf_vi;
x_hat = x_hat_kf_vi;
J = J_kf_vi;
K = K_vi;
L = L_kf;
