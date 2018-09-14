clc
clear all
close all

load P
load T

GOAL = 0.001;

SPREAD2 = 60;

[trainP,valP,testP,trainInd,valInd,testInd]	= dividerand(P, 0.6, 0.2, 0.2);	
[trainT,valT,testT]	= divideind(T,trainInd,valInd,testInd);	

MaxNumNeurons = 1;
NumNeuronsPerDisplay = 1;
currentValidationMSE  = 1000000000000000;
previousValidationMSE = 10000000000000000;

%Stops the second you find a higher validationMSE
while currentValidationMSE < previousValidationMSE
    [net2,tr] = newrb(trainP,trainT,GOAL,SPREAD2,MaxNumNeurons,NumNeuronsPerDisplay);
    MaxNumNeurons= MaxNumNeurons +1;
    previousValidationMSE = currentValidationMSE;
    y1v	= sim(net2,valP);
    currentValidationMSE=mse(y1v-valT)
end

MaxNumNeurons = 1;
a = net2(trainP);
b = net2(valP);
c = net2(testP);

for i = 1:100
    [net2,tr] = newrb(trainP,trainT,GOAL,SPREAD2,MaxNumNeurons,NumNeuronsPerDisplay);
    MaxNumNeurons= MaxNumNeurons +1;
    y1v	= sim(net2,valP);
    currentValidationMSE=mse(y1v-valT);
    NumberOfNodes(i) = i;
    ValMSE(i) = currentValidationMSE;
    TestMSE(i) = perform(net2,testT,i);
    save(['\\kc.umkc.edu\kc-users\home\e\emt9q7\My Documents\MATLAB\NeuralNetsMiniProject2/networks2/net2_' num2str(c) '.mat'],'net2')
end

%save spreadVvalidationVneuro
%load spreadVvalidationVneuron
load net2_52

a = net2(trainP);
b = net2(valP);
c = net2(testP);

plot(NumberOfNodes, ValMSE)
title('Validation MSE vs Neurons vs SPREAD')
xlabel('Neurons')
ylabel('Validation MSE')

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

ezroc3(a,trainTargets);
ezroc3(b,valTargets);
ezroc3(c,testTargets);

% %Extra Work Ignore
% %iterates 100 times for a graph of Validation MSE vs Neurons vs Spread
% %there are 10,000 results I let run from 8am to 1pm only got 2601 results
% %unfortunately I did not get to the higher Spreads (max was 26 spread)
% MaxNumNeurons = 1;
% SPREAD2 = 1;
% a = net2(trainP);
% b = net2(valP);
% c = net2(testP);
% index = 1;
% 
% for i = 1:100
%     MaxNumNeurons = 1;
% for c = 1:100
%     [net2,tr] = newrb(trainP,trainT,GOAL,SPREAD2,MaxNumNeurons,NumNeuronsPerDisplay);
%     MaxNumNeurons= MaxNumNeurons +1;
%     previousValidationMSE = currentValidationMSE;
%     y1v	= sim(net2,valP);
%     currentValidationMSE=mse(y1v-valT);
%     NumberOfNodes(index) = c;
%     Spread(index)= SPREAD2;
%     ValMSE(index) = currentValidationMSE;
%     TestMSE(index) = perform(net2,testT,c);
%     index = index + 1;
% end
%     SPREAD2 = SPREAD2 + 1;
% end
% 
% save spreadVvalidationVneurons
% 
% plot(NumberOfNodes, ValMSE)
% title('Validation MSE vs Neurons vs SPREAD')
% xlabel('Neurons')
% ylabel('Validation MSE')
% 
% plot(Spread, ValMSE)
% title('Validation MSE vs Neurons vs SPREAD')
% xlabel('Spread')
% ylabel('Validation MSE')
% 
% plot3(NumberOfNodes, ValMSE, Spread)
% title('Validation MSE vs Neurons vs SPREAD')
% xlabel('Neurons')
% ylabel('Validation MSE')
% zlabel('Spread')
% 
% NumNodes=[1:100];
% spreadd = [1:27];
% 
% 
% %Change -1 to 0's for the targets
% [m,trainTsize] = size(trainT);
% [m,valTsize] = size(valT);
% [m,testTsize] = size(testT);
% trainTargets = trainT;
% valTargets = valT;
% testTargets = testT;
% 
% %Change train Alternative way (trainTargets+1)/2
% for index = 1:trainTsize
%     i = trainTargets(1,index);
%     if i < 0
%         trainTargets(1,index) = 0;
%     end
% end
% %Change validation
% for index = 1:valTsize
%     i = valTargets(1,index);
%     if i < 0
%         valTargets(1,index) = 0;
%     end
% end
% %Change test
% for index = 1:testTsize
%     i = testTargets(1,index);
%     if i < 0
%         testTargets(1,index) = 0;
%     end
% end
% 
% ezroc3(a,trainTargets);
% ezroc3(b,valTargets);
% ezroc3(c,testTargets);
