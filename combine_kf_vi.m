%% First Scenario : Kalman Filter and Value Iteration 
function [kf_vi] = combine_kf_vi(A,B,C,Q,R,N,L_kf,y)
tic 
[P_vi,K_vi_old,~] = value_iteration(A,N,B,Q,R);
K_vi = K_vi_old;
x_hat_kf_vi = ones(size(A,1),1);
for i = 1:N
    u_kf_vi(:,i) = -K_vi(i,:)*x_hat_kf_vi(:,i);
    y_hat_kf_vi(:,i) = C*x_hat_kf_vi(:,i);
    x_hat_kf_vi(:,i+1) = (A-B*K_vi(i,:))*x_hat_kf_vi(:,i)+L_kf(:,i)*(y_hat_kf_vi(:,i)-y(:,i));         
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
time_elapsed = toc;
kf_vi.P = P;
kf_vi.u = u;
kf_vi.x_hat = x_hat;
kf_vi.y = y;
kf_vi.J = J;
kf_vi.K = K;
kf_vi.L = L;
kf_vi.time = time_elapsed;
