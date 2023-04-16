function KG = KalmanNet(delta_x,target,u,A,C,B)
x_hat_kf = A*delta_x(1:end-1,:)+B*u;
y_hat_kf = C*x_hat_kf;
X = delta_x;
Y = target(1,:);
numFeatures = size(X,1);
numResponses = size(Y,1);
numHiddenUnits = 10;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'StateActivationFunction','tanh')
    fullyConnectedLayer(numResponses)
    regressionLayer];
options = trainingOptions('adam', ...
    'MaxEpochs',1000, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.5, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');
%% Train LSTM Net
net = trainNetwork(X,Y,layers,options);
%% Predict 
for i = 1:N
    nets(i) = predictAndUpdateState(net,X(:,i));
    KG(:,i) = nets(i).Layers(2).HiddenState;
end
% x_hat_next = (KG*Y')+x_hat_kf;
% y_hat = C*x_hat_next;
end
