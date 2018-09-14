clc
clear all
close all
clear

%Initialize Variables
SumTestError = 0;
SumValidationError = 0;
SumTrainingError = 0;

VarianceTestData = [];
VarianceValidationData = [];
VarianceTrainingData = [];

stopPoint = 10;


[x,t] = bodyfat_dataset; %load dataset [inputs, targets]

testInput = x(1:13,203:252); %set aside last 50 samples for testing
testTarget = t(1,203:252);
xnew=x(1:13,1:202);
tnew=t(1,1:202);


%create neural network
net = fitnet(10); %create one hidden layer regression MLP using 'fitnet'
net.divideparam.trainratio = 0.8; %training = 80% of data
net.divideparam.valratio = 0.2; %validation = 20% of data
net.divideparam.testratio = 0; %testing is set aside

%train/test/collect data on neural network
for index = 1:stopPoint

net = init(net) %reset using init????
[net,tr] = train(net,xnew,tnew); %trains the data from bodyfat_dataset

%test the data
y = net(testInput);
z = net(xnew);

%Calculate Targets used/section
validationTargets = tnew .* tr.valMask{1};
trainTargets = tnew .* tr.trainMask{1};

%sum up the test error
SumTestError = SumTestError + perform(net,testTarget,y);
SumValidationError = SumTestError + perform(net,validationTargets,z);
SumTrainError = SumTestError + perform(net,trainTargets,z);

%variance test data
VarianceTestData(index) = [perform(net,testTarget,y)];
VarianceValidationData(index) = [perform(net,validationTargets,z)];
VarianceTrainData(index) = [perform(net,trainTargets,z)];
end

%calculate the mean of MSE
TestMeanMSE = SumTestError/stopPoint
ValidationMeanMSE = SumValidationError/stopPoint
TrainMeanMSE = SumTrainError/stopPoint

%calculate variance of MSE
TestVariance = var(VarianceTestData)
ValidationVariance = var(VarianceValidationData)
TrainVariance = var(VarianceTrainData)

view(net)
%change regularization
plot(t);
hold on, plot(y);




