%% Third Scenario : KalmanNet and VI
function [kn_vi] = combine_kn_vi(A,B,C,Q,R,N,x_hat_net)
tic
[P_vi,K_vi_old,~] = value_iteration(A,N,B,Q,R);
x_hat_kn_vi = x_hat_net;
KK = K_vi_old;
for i = 1:N
    u_kn_vi(1,i) = -KK(i,:)*x_hat_kn_vi(:,i);
    y_hat_kn_vi(i) = C*x_hat_kn_vi(:,i);
    x_hat_kn_vi(:,i+1) = A*x_hat_kn_vi(:,i)+B*u_kn_vi(:,i);
end
% Value function 
J_kn_vi = value_func(x_hat_kn_vi,u_kn_vi,Q,R,N);
% output
P = P_vi;
u = (u_kn_vi);
y = y_hat_kn_vi;
x_hat = abs(x_hat_kn_vi);
J = J_kn_vi;
K = KK;
time_elapsed = toc;
kn_vi.P = P;
kn_vi.u = u;
kn_vi.x_hat = x_hat;
kn_vi.y = y;
kn_vi.J = J;
kn_vi.K = K;
kn_vi.time = time_elapsed;