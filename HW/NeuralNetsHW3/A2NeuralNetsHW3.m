close all;
clear all;
clc;
%//////////////////////////////////INPUT 1///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(1,:)'\d(2,:)'; %Training/MSE linear model creation
y=X(1,:)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,:),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC SUBSET 1, AUC=' num2str(AUC)])

%////////////////////////////////// SUBSET INPUT 1/10///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(1,1:50)'\d(2,1:50)'; %Training/MSE linear model creation
y=X(1,1:50)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,1:50),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TRAINING SUBSET 1, AUC=' num2str(AUC)])

%////////////////////////////////// SUBSET INPUT 2/10///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(1,51:100)'\d(2,51:100)'; %Training/MSE linear model creation
y=X(1,51:100)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,51:100),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TRAINING SUBSET 1, AUC=' num2str(AUC)])

%////////////////////////////////// SUBSET INPUT 3/10///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(1,101:150)'\d(2,101:150)'; %Training/MSE linear model creation
y=X(1,101:150)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,101:150),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TRAINING SUBSET 1, AUC=' num2str(AUC)])

%////////////////////////////////// SUBSET INPUT 4/10///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(1,151:200)'\d(2,151:200)'; %Training/MSE linear model creation
y=X(1,151:200)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,151:200),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TRAINING SUBSET 1, AUC=' num2str(AUC)])

%////////////////////////////////// SUBSET INPUT 5/10///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(1,201:250)'\d(2,201:250)'; %Training/MSE linear model creation
y=X(1,201:250)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,201:250),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TRAINING SUBSET 1, AUC=' num2str(AUC)])

%////////////////////////////////// SUBSET INPUT 6/10///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(1,251:300)'\d(2,251:300)'; %Training/MSE linear model creation
y=X(1,251:300)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,251:300),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TRAINING SUBSET 1, AUC=' num2str(AUC)])

%////////////////////////////////// SUBSET INPUT 7/10///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(1,301:350)'\d(2,301:350)'; %Training/MSE linear model creation
y=X(1,301:350)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,301:350),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TRAINING SUBSET 1, AUC=' num2str(AUC)])

%////////////////////////////////// SUBSET INPUT 8/10///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(1,351:400)'\d(2,351:400)'; %Training/MSE linear model creation
y=X(1,351:400)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,351:400),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TRAINING SUBSET 1, AUC=' num2str(AUC)])

%////////////////////////////////// SUBSET INPUT 9/10///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(1,401:450)'\d(2,401:450)'; %Training/MSE linear model creation
y=X(1,401:450)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,401:450),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TRAINING SUBSET 1, AUC=' num2str(AUC)])

%////////////////////////////////// SUBSET INPUT 10/10///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(1,451:500)'\d(2,451:500)'; %Training/MSE linear model creation
y=X(1,451:500)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,451:500),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TRAINING SUBSET 1, AUC=' num2str(AUC)])

