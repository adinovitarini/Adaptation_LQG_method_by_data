%% Linearize
function [A,B,C] = linearizeModel(net)
nh = net.Layers(2).NumHiddenUnits
Wa = net.Layers(2).RecurrentWeights(1:nh,:);
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
C = Wc*Phi;
end
