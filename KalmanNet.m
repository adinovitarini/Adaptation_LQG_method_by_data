function [KG,net] = KalmanNet(delta_x,target,u,A,B,C,N)
%%  Propagate 
x_hat_kf = A*delta_x(1:end-1,:)+B*u;
y_hat_kf = C*x_hat_kf;
X = delta_x;
Y = target(1:end-1,:);
tic
numFeatures = size(X,1);
numResponses = size(Y,1);
numHiddenUnits = 3;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'StateActivationFunction','tanh','GateActivationFunction','sigmoid')
    dropoutLayer(0.9,"Name","dropout")
    fullyConnectedLayer(numResponses)
    regressionLayer];
options = trainingOptions('adam', ...
    'MaxEpochs',100, 'MiniBatchSize',8,...
    'GradientThreshold',.1, ...
    'InitialLearnRate',0.015, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',1, ...
    'Plots','none');
%% Train LSTM Net
net = trainNetwork(X,Y,layers,options);
%% Predict 
net = predictAndUpdateState(net,X);
[net,KG] = predictAndUpdateState(net,target);
time_elapsed = toc;
fprintf('MSE : %f\n',mse(KG,target(1,:)));
