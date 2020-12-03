
clear all
close all

% Load the dataset
load ModelData
TrainSixFold = Final_Combined_Fold;
TestSixFold = Fold;
%AA_position = ["A";"R";"N";"D";"C";"Q";"E";"G";"H";"I";"L";"K";"M";"F";"P";"S";"T";"W";"Y";"V"];

cd 'C:\Users\User\Desktop\libsvm-weights-3.22\matlab'
% Change directory to where libsvm is saved. e.g. cd 'C:\Users\User\Desktop\libsvm-weights-3.22\matlab'

for i = 1:6 % get the test and train datasets (6 fold cross validation)

    Train_Data = TrainSixFold{i};
    for size_train = 1:size(Train_Data,1)
        
        % Select the features which have not been eliminated using backward elimination method
        Train_Data{size_train,5} = [Train_Data{size_train,5}(6,:);Train_Data{size_train,5}(8,:);Train_Data{size_train,5}(12,:);Train_Data{size_train,5}(15,:);Train_Data{size_train,5}(10,:);Train_Data{size_train,5}(1,:);Train_Data{size_train,5}(16,:);Train_Data{size_train,5}(2,:);Train_Data{size_train,5}(19,:);Train_Data{size_train,5}(9,:);Train_Data{size_train,5}(5,:);Train_Data{size_train,5}(3,:);Train_Data{size_train,5}(18,:);Train_Data{size_train,5}(13,:)];
        Train_Data{size_train,2} = transpose(Train_Data{size_train,5}(:)); % rearrange to get vector 
        vec = cell2mat(Train_Data(size_train,2));
        [M,I] = max(vec);
        Train_Data{size_train,2} = Train_Data{size_train,2}/M;
        
    end
    Train = cell2mat(Train_Data(:,2)); % Get the Train dataset features
    train_label = cell2mat(Train_Data(:,3)); % Get the label (originally in string)
    Train_label = str2num(train_label); % Get the Train dataset labels
    
    Test_Data = TestSixFold{i};
    for size_test = 1:size(Test_Data,1)
        
        % Select the features which have not been eliminated using backward elimination method
        Test_Data{size_test,5} = [Test_Data{size_test,5}(6,:);Test_Data{size_test,5}(8,:);Test_Data{size_test,5}(12,:);Test_Data{size_test,5}(15,:);Test_Data{size_test,5}(10,:);Test_Data{size_test,5}(1,:);Test_Data{size_test,5}(16,:);Test_Data{size_test,5}(2,:);Test_Data{size_test,5}(19,:);Test_Data{size_test,5}(9,:);Test_Data{size_test,5}(5,:);Test_Data{size_test,5}(3,:);Test_Data{size_test,5}(18,:);Test_Data{size_test,5}(13,:)];
        Test_Data{size_test,2} = transpose(Test_Data{size_test,5}(:)); % rearrange to get vector 
        vec = cell2mat(Test_Data(size_test,2));
        [M,I] = max(vec);
        Test_Data{size_test,2} = Test_Data{size_test,2}/M;
        
    end
    Test = cell2mat(Test_Data(:,2)); % Get the Test dataset features
    test_label = cell2mat(Test_Data(:,3)); % Get the label (originally in string)
    Test_label = str2num(test_label); % Get the Test dataset labels

    % Train LibSVM-weights. Empty square brackets denote we are not supplying weights
    model=svmtrain([],Train_label,Train,['-s 0 -t 0 -b 1']); % -s 0; svm type = C-SVC, -t 0; kernel = linear
    
    % Test the classifier
    [pred,acc,prob_values]=svmpredict(Test_label,Test,model,'-b 1');
    
    % Save the probability estimates and the true labels
    prediction{i} = prob_values(:,1);
    True_label{i} = Test_label;

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
            
    Results(1,i) = sen; 
    Results(2,i) = spe; 
    Results(3,i) = pre; 
    Results(4,i) = accuracy; 
    Results(5,i) = mcc;
end
Results_Avg = sum(Results,2)/6; % Average the result to get 6 fold CV 

% AUC calculation
classes = [True_label{1}; True_label{2}; True_label{3}; True_label{4}; True_label{5}; True_label{6}];
scores = [prediction{1}; prediction{2}; prediction{3}; prediction{4}; prediction{5}; prediction{6}];

[X1,Y1,T,AUC_1] = perfcurve(classes,scores,1);

% Display the result
metrics = {'sensitivity'; 'specificity'; 'precision'; 'accuracy'; 'mcc'};
Six_Fold_CV = [metrics, num2cell(Results_Avg)] 
AUC_1


% Plot of ROC Curve
plot(X1,Y1,'linewidth',3)
title('ROC Curve')
xlabel('False Positive Rate')
ylabel('True Positive Rate')

