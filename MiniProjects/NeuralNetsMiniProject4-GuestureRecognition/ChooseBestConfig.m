clc
clear all
close all

%Best Configuration => (Delay 20, Node 15)
%Runner Up Configuration => (Delay 20, Node 10)
%3rd place Configuration => (Delay 20, Node 5)
addpath(fullfile('\\kc.umkc.edu\kc-users\home\e\emt9q7\My Documents\MATLAB\NeuralNetsMiniProject4/PersonX/')); 
load('Person_2.mat');

Train = [Round1Circle; Round1Triangle; Round1Right; Round1Down]; %1200x2
Train = Train'; %2x1200
Train = con2seq(Train);

Test = [Round2Circle; Round2Triangle; Round2Right; Round2Down]; %1200x2
Test = Test'; %2x1200
Test = con2seq(Test);

%Target Values
%Circle targets
CTargets = [ones(1,300); -ones(1,300); -ones(1,300); -ones(1,300)];
%Triangle targets
TTargets = [-ones(1,300); ones(1,300); -ones(1,300); -ones(1,300)];
%Right targets
RTargets = [-ones(1,300); -ones(1,300); ones(1,300); -ones(1,300)];
%Down targets
DTargets = [-ones(1,300); -ones(1,300); -ones(1,300); ones(1,300)];

Targets = [CTargets TTargets RTargets DTargets]; 
Targets = [Targets];
Targets = con2seq(Targets);

I = Train;

T = Targets;


%////////////////////////////end data///////////////////////////////////

%Create the time delay network
inputDelay = 1:30; % 25 30
hiddenSize = 15; % 10 15
outputDelay = 1:30; %10 15 20
for outputDelaysize = 10:5:20
    outputDelay = 1:outputDelaysize; %10 15 20
for inputDelaysize = 25:5:30
    inputDelay = 1:inputDelaysize; %25 30
for hiddenNodeSize = 10:5:15
    hiddenSize = hiddenNodeSize; %10 15
Htotal = [];
H = [];


for k = 1:5
net = narxnet(inputDelay,outputDelay,hiddenSize); %create network
net.layers{2}.transferFcn='tansig';
net.divideparam.trainratio = 1; %training = 100% of data
net.divideparam.valratio = 0;
net.divideparam.testratio = 0; 
net.trainFcn='trainbr'; % use Bayesian regularization backpropagation
net.trainParam.epochs = 100; % 100 epochs
[Xs,Xi,Ai,Ts] = preparets(net,I,{},T);

[net,tr] = train(net,Xs,Ts,Xi,Ai); %train network

%close the loop before testing
netc = closeloop(net);
netc.name = [net.name ' - Closed Loop'];
[Xc,Xic,Aic,Tc] = preparets(netc,I,{},T);
y2 = netc(Xc,Xic,Aic);
closedLoopPerformance = perform(netc,Tc,y2);

%test network
testNet = netc(Test, Xic, Aic); %??????????????????????????????????wut

%view(netc)

testNet = cell2mat(testNet);

%Calculate Classifier output matrices

H = [];

for i = 1:4
    for j = 1:4
        H(j,i) = sum(testNet(i,300*(j-1)+1:300*j))/(size(testNet,2)/4); %size of testnet/4 = 300
    end
end
Htotal = cat(3,Htotal,H);
net = init(net);%initialize network
end
%ezroc3(cat(3,H1,H2,H3));
ezroc3(Htotal);
%savefig
%savefig(['\\kc.umkc.edu\kc-users\home\e\emt9q7\My Documents\MATLAB\NeuralNetsMiniProject4/BestConfigROCFigureX/FigureInput_' num2str(inputDelaysize) '_Output_' num2str(outputDelaysize) '_Node_' num2str(hiddenNodeSize) '.fig']);

end
end
end




