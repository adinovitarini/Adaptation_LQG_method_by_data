function [Model,x,y] = HaykinModel(net,consig)
nh = net.Layers(2).NumHiddenUnits;
Rw = net.Layers(2).RecurrentWeights;
for i = 1:4
w(:,:,i) = Rw(((i-1)*nh)+1:(i*nh),:);
end
Wa = w(:,:,3);
Wb = net.Layers(2).InputWeights(1:nh,1);
Wc = net.Layers(3).Weights;
Model.A = Wa;
Model.B = Wb;
Model.C = Wc;
disp('The nonlinearity function : \n');
disp('1. Sigmoid')
disp('2. Tanh')
j = input('Choose the nonlinearity function for Haykin Model : \n');
x = zeros(nh,1);
u = consig;
if j==1
    for i = 1:size(u,1)
    x(:,i+1) = logsig(Wa*x(:,i)+Wb*u(i));
    y(i) = (Wc*x(:,i));
    end
else if j==2
    for i = 1:size(u,1)
    x(:,i+1) = tanh(Wa*x(:,i)+Wb*u(i));
    y(i) = (Wc*x(:,i));
    end
    end
end