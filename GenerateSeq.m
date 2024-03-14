function dataset = GenerateSeq(sys,N,Rww,Rvv,Q,R)
% This code is used for generate the sequence the estimated state that
% obtain from KF and LQR 
[L,~] = KalmanConventional(sys,N,Rww,Rvv);
[K,~,~] = lqr(sys,Q,R);
x = zeros(size(sys.A,1),N);
x_hat = x;
x_nw = x;
y = zeros(size(sys.B,2),N);
y_hat = y;
y_nw = y;
w = wgn(N,1,1);
v = w;
tic
for i = 1:N
    x(:,i+1) = (sys.A-sys.B*K)*x(:,i)+w(i);
    y(i) = (sys.C*x(:,i))+v(i);
    x(:,i) = x(:,i+1);
    x_hat(:,i+1) = ((sys.A-sys.B*K)*x_hat(:,i))+L(i)*(y(i)-y_hat(i));
    y_hat(i) = sys.C*x_hat(:,i);
    x_hat(:,i) = x_hat(:,i+1);
    x_nw(:,i+1) = (sys.A-sys.B*K)*x_nw(:,i);
    y_nw(i) = (sys.C*x_nw(:,i));
    x_nw(:,i) = x_nw(:,i+1);
end
time_elapsed = toc;
u = -K*x_hat;
dataset.x_hat = x_hat;
dataset.x_nw = x_nw;
dataset.x = x;
dataset.y_hat = y_hat;
dataset.y_nw = y_nw;
dataset.y = y;
dataset.u = u;
dataset.L = L;
dataset.K = K;
dataset.time = time_elapsed;