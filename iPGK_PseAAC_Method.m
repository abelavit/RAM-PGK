
clear all
close all

% Load the dataset and iPGK_PseAAC feature set
load ModelData
load iPGK_Data

TrainSixFold = Final_Combined_Fold;
TestSixFold = Fold;

for i = 1:6
    iPGK_Test{i} = Get_iPGK(TestSixFold{i}, iPGK_Data); % Get the iPGK_PseAAC features for the test dataset
    
    iPGK_Train{i} = Get_iPGK(TrainSixFold{i}, iPGK_Data); % Get the iPGK_PseAAC features for the train dataset
    
end
tic
cd 'C:\Users\User\Desktop\libsvm-weights-3.22\matlab'
% Change directory to where libsvm is saved. e.g. cd 'C:\Users\User\Desktop\libsvm-weights-3.22\matlab'

for i = 1:6 % get the test and train datasets (6 fold cross validation)
   
    Test_Data = iPGK_Test{i}; 
    Test = cell2mat(Test_Data(:,2)); % Get the features
    test_label = cell2mat(Test_Data(:,3)); % Get the label (originally in string)
    Test_label = str2num(test_label); % Change string label to num
    
    Train_Data = iPGK_Train{i}; 
    Train = cell2mat(Train_Data(:,2)); % Get the features
    train_label = cell2mat(Train_Data(:,3)); % Get the label (originally in string)
    Train_label = str2num(train_label); % Change string label to num
    
    % Train LibSVM-weights. Parameters are same as the iPGK_PseAAC predictor. Empty square brackets denote we are not supplying weights
    model=svmtrain([],Train_label,Train,['-s 0 -t 0 -c 100 -g 0.675 -b 1 -h 0']);
    
    % Test the classifier
    [predinitial,acc,prob_values]=svmpredict(Test_label,Test,model,'-b 1');
    
    % Save the probability estimates and the true labels
    prediction{i} = prob_values(:,1);
    %prediction{i} = pred;
    True_label{i} = Test_label;
    
    % cutoff ? of 0.45. Same as the iPGK_PseAAC predictor
    for prob_index = 1:size(prob_values,1)
        
        if prob_values(prob_index) > 0.45
            
            pred(prob_index) = 1;
            
        else
        
            pred(prob_index) = 0; 
            
        end
    end

    % Obtaining the FN, FP, TN and TP values
    FN = 0;
    FP = 0;
    TN = 0;
    TP = 0;
    for j = 1:size(Test_label,1)
        if Test_label(j) == 1
            if pred(j) == 1
                TP = TP + 1;
            else
                FN = FN + 1; 
            end
        else
            if pred(j) == 1
                FP = FP + 1;
            else
                TN = TN + 1; 
            end
        end
    end
    
    % Calculating the performance metrics
    sen = TP/(TP+FN);
            check_sen = isnan(sen); % Check if NaN

            if check_sen == true
                sen = 0; % If NaN
            end
    spe = TN/(TN+FP);
            check_spe = isnan(spe); % Check if NaN

            if check_spe == true
                spe = 0; % If NaN
            end
    pre = TP/(TP+FP);
            check_pre = isnan(pre); % Check if NaN

            if check_pre == true
                pre = 0; % If NaN
            end
    accuracy = (TN+TP)/(FN+FP+TN+TP);
            check_acc = isnan(accuracy); % Check if NaN

            if check_acc == true
                accuracy = 0; % If NaN
            end
    mcc = ((TN*TP)-(FN*FP))/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
            check_mcc = isnan(mcc); % Check if NaN

            if check_mcc == true
                mcc = -1; % If NaN
            end
    Results(:,i) = [sen;spe;pre;accuracy;mcc]; 
    
    % Save all the predictions for each fold
    PredIntital(:,i) = predinitial;
    Probvalues{:,i} = prob_values;
    PredFinal(:,i) = pred;
end
Results_Avg = sum(Results,2)/6; % Average the result to get 6 fold CV
toc
% AUC calculation
classes = [True_label{1}; True_label{2}; True_label{3}; True_label{4}; True_label{5}; True_label{6}];
scores = [prediction{1}; prediction{2}; prediction{3}; prediction{4}; prediction{5}; prediction{6}];

[X3,Y3,T,AUC_3] = perfcurve(classes,scores,1);

% Display the result
metrics = {'sensitivity'; 'specificity'; 'precision'; 'accuracy'; 'mcc'};
Six_Fold_CV = [metrics, num2cell(Results_Avg)] 
AUC_3

% Plot of ROC Curve
plot(X3,Y3,'linewidth',3)
title('ROC Curve')
xlabel('False Positive Rate')
ylabel('True Positive Rate')
