clc
clear all
close all

Htotal = []; %Final H value used for ROC curve
Hmatrix = []; % H value that will hold H1,H2,H3....H(crossfold)
Havg = []; %average of all values in Hmatrix (used for Extra Credit ROC)
crossfold = 6; %define crossfold
inputDelay = 1:20;%use best delay
hiddenSize = 15;%use best node size
BestDelay = 2;
BestSize = 1;
personNum = 1;
crossNum = 5;
personX = [] %person matrix
%populate personX matrix
for personNum = 1:6
    addpath(fullfile('\\kc.umkc.edu\kc-users\home\e\emt9q7\My Documents\MATLAB\NeuralNetsMiniProject3/PersonX/')); 
    personFileName = sprintf('Person_%d.mat',personNum)
    load(personFileName);
    getPersonData = [Round1Circle' Round1Triangle' Round1Right' Round1Down' Round2Circle' Round2Triangle' Round2Right' Round2Down' Round3Circle' Round3Triangle' Round3Right' Round3Down'];
    %fill person matrix
    personX = cat(1,personX,getPersonData);
end

%for inputdelay
%for hiddenSize
%calc Havg
%Hmatrix <= Havg    

%personX=> person 1 = 1:2, person 2 = 3:4, person 3 = 5:6, person 4 = 7:8
%person 5 = 9:10, person 6 = 11:12
for crossNum = 1:crossfold
%Crossfold Conditions
if crossNum == 1 
Train = [personX(1:2,1:3600) personX(3:4,1:3600) personX(5:6,1:3600) personX(7:8,1:3600) personX(11:12,1:3600)]; %2x18000
Test = personX(11:12,1:3600); %2x3600
end
if crossNum == 2
Train = [personX(1:2,1:3600) personX(3:4,1:3600) personX(5:6,1:3600) personX(7:8,1:3600) personX(11:12,1:3600)]; %2x18000
Test = personX(9:10,1:3600); %2x3600
end
if crossNum == 3
Train = [personX(1:2,1:3600) personX(3:4,1:3600) personX(5:6,1:3600) personX(11:12,1:3600) personX(9:10,1:3600)]; %2x18000
Test = personX(7:8,1:3600); %2x3600
end
if crossNum == 4
Train = [personX(1:2,1:3600) personX(3:4,1:3600) personX(11:12,1:3600) personX(7:8,1:3600) personX(9:10,1:3600)]; %2x18000
Test = personX(5:6,1:3600); %2x3600
end
if crossNum == 5
Train = [personX(1:2,1:3600) personX(11:12,1:3600) personX(5:6,1:3600) personX(7:8,1:3600) personX(9:10,1:3600)]; %2x18000
Test = personX(3:4,1:3600); %2x3600
end
if crossNum == 6
Train = [personX(11:12,1:3600) personX(3:4,1:3600) personX(5:6,1:3600) personX(7:8,1:3600) personX(9:10,1:3600)]; %2x18000
Test = personX(1:2,1:3600); %2x3600
end


%Invert and convert data
Train = con2seq(Train);%2x18000
Test = con2seq(Test);%2x3600

%Create Target Values
%Circle targets
CTargets = [ones(1,300); -ones(1,300); -ones(1,300); -ones(1,300)];
%Triangle targets
TTargets = [-ones(1,300); ones(1,300); -ones(1,300); -ones(1,300)];
%Right targets
RTargets = [-ones(1,300); -ones(1,300); ones(1,300); -ones(1,300)];
%Down targets
DTargets = [-ones(1,300); -ones(1,300); -ones(1,300); ones(1,300)];

Targets = [CTargets TTargets RTargets DTargets]; 
Targets = [Targets Targets Targets Targets Targets Targets Targets Targets Targets Targets Targets Targets Targets Targets Targets ];
Targets = con2seq(Targets);

I = Train;

T = Targets;

%////////////////////////////end data///////////////////////////////////

%Create the time delay network
net = timedelaynet(inputDelay,hiddenSize); %create network
net.layers{2}.transferFcn='tansig'; %change to tansig function
net.divideparam.trainratio = 1; %training = 100% of data 
net.divideparam.valratio = 0; %disable early stopping
net.divideparam.testratio = 0; 
net.trainFcn='trainbr'; % use Bayesian regularization backpropagation
net.trainParam.epochs = 100; % 100 epochs
[Xs,Xi,Ai,Ts] = preparets(net,I,T);

net = init(net);%initialize network
[net,tr] = train(net,Xs, Ts); %train network

%test network
testNet = net(Test, Xi, Ai);

testNet = cell2mat(testNet); %convert test output from cell to matrix

%Calculate Classifier output matrices
H=[];
for i=1:4
H(i,1)= (sum(testNet(i,1:300))+sum(testNet(i,1201:1500))+sum(testNet(i,2401:2700)))/900;
H(i,2)=  (sum(testNet(i,301:600))+sum(testNet(i,1501:1800))+sum(testNet(i,2701:3000)))/900;
H(i,3)=  (sum(testNet(i,601:900))+sum(testNet(i,1801:2100))+sum(testNet(i,3001:3300)))/900;
H(i,4)=  (sum(testNet(i,901:1200))+sum(testNet(i,2101:2400))+sum(testNet(i,3301:3600)))/900;
end

% 
% 
% testNetwork = 0;
% 
% testNet1 = testNet(:,1:1200);
% testNet2 = testNet(:,1201:2400);
% testNet3 = testNet(:,2401:3600);
% Hsum= zeros(4,4);
% HfinalAvg = [];
% 
% for testnetNum=1:1
%     if testnetNum == 1
%         testNetwork = testNet1;
%     end
%     if testnetNum == 2
%         testNetwork = testNet2;
%     end
%     if testnetNum == 3
%         testNetwork = testNet3;
%     end
%     for i = 1:4
%         for j = 1:4
%          H(j,i) = sum(testNetwork(i,300*(j-1)+1:300*j))/(size(testNetwork,2)/4);
%         end
%     end
%     
%     Hsum = Hsum + H;
% end
% 
% HfinalAvg = Hsum/3
% 
% %   for j = 1:4
% %       for i = 1:4
% %           H(j,i) = (sum(testNet(j,(300*(i-1)+1):300*i))+sum(testNet(j,(1201+300*(i-1)+1):(1201+300*i)))+sum(testNet(j,(2401+300*(i-1)+1):(2401+300*i))))/(size(testNet,2)/4);
% %       end
% %   end
% 
% %2x3600
% %intervals 1:1200 1201:2400 2401:3600
 Htotal = cat(3,Htotal,H);

%save net??????????????????????????????????????????????????????????????????
%save (['\\kc.umkc.edu\kc-users\home\e\emt9q7\My Documents\MATLAB\NeuralNetsMiniProject3/SubjectIndependentNets/NetDelay_' num2str(BestDelay) '_Node_' num2str(BestSize) '_Person_' num2str(personNum) '_fold_' num2str(crossNum) '.mat'], 'net');


end

%ROC Curve
%save(Htotal);

ezroc3(Htotal);
%savefig(['\\kc.umkc.edu\kc-users\home\e\emt9q7\My Documents\MATLAB\NeuralNetsMiniProject3/6FoldROCCurves/FigureDelay_' num2str(BestDelay) '_Node_' num2str(BestSize) '_Person_' num2str(personNum) '.fig']);





%ezroc3(Havg);
%networks = 6 people x 3 folds = 18 saved networks
%netwroks = 6 folds = 6 saved networks
