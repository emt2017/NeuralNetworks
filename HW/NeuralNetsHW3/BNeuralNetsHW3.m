%/////////////////////////////////////(1/2) TRAINING DATA////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(1:9,1:350)'\d(2,1:350)'; %Training/MSE linear model creation
y=X(1:9,1:350)'*w; %Activation/testing (we'll talk about big no no here later)


%ezroc3(X(3,:),d(2,:));
[X,Y,T,AUC] = perfcurve(d(2,1:350),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TRAINING DATA, AUC=' num2str(AUC)])

%/////////////////////////////////////(1/2) TESTING DATA///////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(1:9,351:699)'\d(2,351:699)'; %Training/MSE linear model creation
y=X(1:9,351:699)'*w; %Activation/testing (we'll talk about big no no here later)


%ezroc3(X(3,:),d(2,:));
[X,Y,T,AUC] = perfcurve(d(2,351:699),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TESTING DATA, AUC=' num2str(AUC)])

%//////////////////////////////////TRAINING SUBSET 1///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(1,1:350)'\d(2,1:350)'; %Training/MSE linear model creation
y=X(1,1:350)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,1:350),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TRAINING SUBSET 1, AUC=' num2str(AUC)])

%//////////////////////////////////TEST SUBSET 1///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(1,351:699)'\d(2,351:699)'; %Training/MSE linear model creation
y=X(1,351:699)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,351:699),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TEST SUBSET 1, AUC=' num2str(AUC)])

%//////////////////////////////////TRAINING SUBSET 2///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(2,1:350)'\d(2,1:350)'; %Training/MSE linear model creation
y=X(2,1:350)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,1:350),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TRAINING SUBSET 2, AUC=' num2str(AUC)])

%//////////////////////////////////TEST SUBSET 2///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(2,351:699)'\d(2,351:699)'; %Training/MSE linear model creation
y=X(2,351:699)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,351:699),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TEST SUBSET 2, AUC=' num2str(AUC)])

%//////////////////////////////////TRAINING SUBSET 3///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(3,1:350)'\d(2,1:350)'; %Training/MSE linear model creation
y=X(3,1:350)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,1:350),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TRAINING SUBSET 3, AUC=' num2str(AUC)])

%//////////////////////////////////TEST SUBSET 3///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(3,351:699)'\d(2,351:699)'; %Training/MSE linear model creation
y=X(3,351:699)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,351:699),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TEST SUBSET 3, AUC=' num2str(AUC)])
%//////////////////////////////////TRAINING SUBSET 4///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(4,1:350)'\d(2,1:350)'; %Training/MSE linear model creation
y=X(4,1:350)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,1:350),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TRAINING SUBSET 4, AUC=' num2str(AUC)])

%//////////////////////////////////TEST SUBSET 4///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(4,351:699)'\d(2,351:699)'; %Training/MSE linear model creation
y=X(4,351:699)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,351:699),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TEST SUBSET 4, AUC=' num2str(AUC)])
%//////////////////////////////////TRAINING SUBSET 5///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(5,1:350)'\d(2,1:350)'; %Training/MSE linear model creation
y=X(5,1:350)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,1:350),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TRAINING SUBSET 5, AUC=' num2str(AUC)])

%//////////////////////////////////TEST SUBSET 5///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(5,351:699)'\d(2,351:699)'; %Training/MSE linear model creation
y=X(5,351:699)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,351:699),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TEST SUBSET 5, AUC=' num2str(AUC)])
%//////////////////////////////////TRAINING SUBSET 6///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(6,1:350)'\d(2,1:350)'; %Training/MSE linear model creation
y=X(6,1:350)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,1:350),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TRAINING SUBSET 6, AUC=' num2str(AUC)])

%//////////////////////////////////TEST SUBSET 6///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(6,351:699)'\d(2,351:699)'; %Training/MSE linear model creation
y=X(6,351:699)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,351:699),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TEST SUBSET 6, AUC=' num2str(AUC)])
%//////////////////////////////////TRAINING SUBSET 7///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(7,1:350)'\d(2,1:350)'; %Training/MSE linear model creation
y=X(7,1:350)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,1:350),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TRAINING SUBSET 7, AUC=' num2str(AUC)])

%//////////////////////////////////TEST SUBSET 7///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(7,351:699)'\d(2,351:699)'; %Training/MSE linear model creation
y=X(7,351:699)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,351:699),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TEST SUBSET 7, AUC=' num2str(AUC)])
%//////////////////////////////////TRAINING SUBSET 8///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(8,1:350)'\d(2,1:350)'; %Training/MSE linear model creation
y=X(8,1:350)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,1:350),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TRAINING SUBSET 8, AUC=' num2str(AUC)])

%//////////////////////////////////TEST SUBSET 8///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(8,351:699)'\d(2,351:699)'; %Training/MSE linear model creation
y=X(8,351:699)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,351:699),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TEST SUBSET 8, AUC=' num2str(AUC)])
%//////////////////////////////////TRAINING SUBSET 9///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(9,1:350)'\d(2,1:350)'; %Training/MSE linear model creation
y=X(9,1:350)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,1:350),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TRAINING SUBSET 9, AUC=' num2str(AUC)])

%//////////////////////////////////TEST SUBSET 9///////////////////////////////////
[X,d] = cancer_dataset; %Type help cancer_dataset for more info
%Transposes X(1-9 attributes) then /=(X^(-1))
% * the transpose of 2nd row of d i.e. w = X^(-1)*d...
w=X(9,351:699)'\d(2,351:699)'; %Training/MSE linear model creation
y=X(9,351:699)'*w; %Activation/testing (we'll talk about big no no here later)
[X,Y,T,AUC] = perfcurve(d(2,351:699),y',1);

figure,plot(X,Y) %Visualize
xlabel('False positive rate')
ylabel('True positive rate')
title(['2D ROC TEST SUBSET 9, AUC=' num2str(AUC)])

