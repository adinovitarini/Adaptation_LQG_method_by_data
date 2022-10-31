function [Model,x,y] = linearizeModel(net,consig)
nh = net.Layers(2).NumHiddenUnits;
Rw = net.Layers(2).RecurrentWeights;
for i = 1:4
w(:,:,i) = Rw(((i-1)*nh)+1:(i*nh),:);
end
Wa = w(:,:,3);
Wb = net.Layers(2).InputWeights(1:nh,1);
Wc = net.Layers(3).Weights;
disp('The nonlinearity function : \n');
disp('1. Sigmoid')
disp('2. Tanh')
j = input('Choose the nonlinearity function : \n');
if j==1
    Phi = 0.25*eye(nh,nh);
else if j==2
    Phi = eye(nh,nh);
end
end
A = Phi*Wa;
B = Phi*Wb;
C = Wc;
Model.A = A;
Model.B = B;
Model.C = C;
x = zeros(nh,1);
u = consig;
if j==1
    for i = 1:size(u,1)
    x(:,i+1) = logsig(Wa*x(:,i)+Wb*u(i));
    y(i) = Wc*x(:,i);
    end
else if j==2
    for i = 1:size(u,1)
    x(:,i+1) = tanh(Wa*x(:,i)+Wb*u(i));
    y(i) = Wc*x(:,i);
    end
    end
end
