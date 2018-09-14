clc
clear all
close all

load P
load T

%Change the SPREAD
SPREAD1 = 35;
SPREAD2 = 60;

%Change the GOAL
GOAL = 0.5;

%Divide Sets
[trainP,valP,testP,trainInd,valInd,testInd]	= dividerand(P, 0.6, 0.2, 0.2);	
[trainT,valT,testT]	= divideind(T,trainInd,valInd,testInd);	

%Train Neural Networks with radial basis function
net1	=	newrbe(trainP,trainT,SPREAD1); %exact
net2	=	newrb(trainP,trainT,GOAL,SPREAD2);

%load pre-trained network
load netrbe

%Find size
net1.layers{1}
net2.layers{1}

%Find validation mse
y1v	= sim(net1,valP);
mse1v=mse(y1v-valT)

y1v	= sim(net2,valP);
mse2v=mse(y1v-valT)

%Test network
a = net1(trainP);
b = net1(valP);
c = net1(testP);

w = net2(trainP);
y = net2(valP);
z = net2(testP);

%Find Test MSE
TestMSE1 = perform(net1,testT,c)
TestMSE2 = perform(net2,testT,z)

%Change -1 to 0's for the targets
[m,trainTsize] = size(trainT);
[m,valTsize] = size(valT);
[m,testTsize] = size(testT);
trainTargets = trainT;
valTargets = valT;
testTargets = testT;

%Change train Alternative way (trainTargets+1)/2
for index = 1:trainTsize
    i = trainTargets(1,index);
    if i < 0
        trainTargets(1,index) = 0;
    end
end
%Change validation
for index = 1:valTsize
    i = valTargets(1,index);
    if i < 0
        valTargets(1,index) = 0;
    end
end
%Change test
for index = 1:testTsize
    i = testTargets(1,index);
    if i < 0
        testTargets(1,index) = 0;
    end
end

%Plot training,validation,test confusion matrices ???????????
%net1
figure 
plotconfusion(trainTargets,(sign(a)+1)/2)
title('Training')
figure 
plotconfusion(valTargets,(sign(b)+1)/2)
title('Validation')
figure 
plotconfusion(testTargets,(sign(c)+1)/2)
title('Test')
%net2
figure 
plotconfusion(trainTargets,(sign(w)+1)/2)
title('Training')
figure 
plotconfusion(valTargets,(sign(y)+1)/2)
title('Validation')
figure 
plotconfusion(testTargets,(sign(z)+1)/2)
title('Test')

%Plot training,validation,test ROC Curves
%net1
ezroc3(a,trainTargets);
ezroc3(b,valTargets);
ezroc3(c,testTargets);
%net2
ezroc3(w,trainTargets);
ezroc3(y,valTargets);
ezroc3(z,testTargets);

% Alternative to plot ROC curves
% [X1,Y1,T1,AUC]=perfcurve(trainT,w,1);
% plot(X1,Y1)
% AUC

