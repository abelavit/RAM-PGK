
% Compare and extract similar Testing data from CKSAAP_Data. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all 

% Load the dataset and CKSAAP_PhoglySite feature set
load ModelData
load CKSAAP_Data % This is the dataset containing CKSAAP features for each lysine K
load Feature_Label

TrainSixFold = Final_Combined_Fold;
TestSixFold = Fold;

for i = 1:6
    CKSAAP_Test{i} = Get_CKSAAP(TestSixFold{i}, CKSAAP_Data); % Get the iPGK_PseAAC features for the test dataset
    
    CKSAAP_Train{i} = Get_CKSAAP(TrainSixFold{i}, CKSAAP_Data); % Get the iPGK_PseAAC features for the train dataset
    
end
tic
%F_Score calculation for each training set
for m = 1:6
    % Number of +ve and -ve samples and Separating +ve and -ve samples
    % The if conditions below will result in error if sample label is of type
    % double
    Data2 = CKSAAP_Train{m};

    Num_of_Pos  = 0;
    Num_of_Neg  = 0;
    for i = 1:size(Data2,1)
        if Data2{i,3} == '1'
           Num_of_Pos = Num_of_Pos + 1;
           Pos_sample(Num_of_Pos,:) = Data2(i,:); 
        end
        if Data2{i,3} == '0'
           Num_of_Neg = Num_of_Neg + 1;
           Neg_sample(Num_of_Neg,:) = Data2(i,:);
        end    
    end
    % Sum the columns of the Final_Data
    for i = 5:size(Data2,2)

        Summ = 0;
        for j = 1:size(Data2,1)

          Summ = Summ + Data2{j,i};  

        end
        Feature_sum{1,i} = Summ;
    end

    % Mean j-th feature in whole
    for i = 5:size(Feature_sum, 2)

         Feature_mean_whole{1,i} = Feature_sum{1,i}/size(Data2,1);    

    end

    % Mean j-th feature in positive

    % Sum the columns of the Pos_sample
    for i = 5:size(Pos_sample,2)

        Summ = 0;
        for j = 1:size(Pos_sample,1)

          Summ = Summ + Pos_sample{j,i};  

        end
        Feature_sum_pos{1,i} = Summ;
    end
    % Mean j-th feature in Pos_sample
    for i = 5:size(Feature_sum_pos, 2)

         Feature_mean_pos{1,i} = Feature_sum_pos{1,i}/size(Pos_sample,1);    

    end

    % Mean j-th feature in negative

    % Sum the columns of the Neg_sample
    for i = 5:size(Neg_sample,2)

        Summ = 0;
        for j = 1:size(Neg_sample,1)

          Summ = Summ + Neg_sample{j,i};  

        end
        Feature_sum_neg{1,i} = Summ;
    end
    % Mean j-th feature in Neg_sample
    for i = 5:size(Feature_sum_neg, 2)

         Feature_mean_neg{1,i} = Feature_sum_neg{1,i}/size(Neg_sample,1);    

    end

    % Denominator part of F_Score formula

    % Pos part of deno
    for i = 5:size(Pos_sample, 2)
        Summ = 0;
        for j = 1:size(Pos_sample, 1)

            Summ = Summ + (Pos_sample{j,i} - Feature_mean_pos{1,i})^2;

        end
        Pos_deno{1,i} = (1/(size(Pos_sample, 1) - 1)) * Summ;
    end
    % Neg part of deno
    for i = 5:size(Neg_sample, 2)
        Summ = 0;
        for j = 1:size(Neg_sample, 1)

            Summ = Summ + (Neg_sample{j,i} - Feature_mean_neg{1,i})^2;

        end
        Neg_deno{1,i} = (1/(size(Neg_sample, 1) - 1)) * Summ;
    end

    % F-Score Calculation

    for i = 5:size(Data2, 2)

        if (Pos_deno{1,i} + Neg_deno{1,i}) == 0
            F_Score{1,i} = 0;

        else
            F_Score{1,i} = ((Feature_mean_pos{1,i}-Feature_mean_whole{1,i})^2 + (Feature_mean_neg{1,i}-Feature_mean_whole{1,i})^2)/(Pos_deno{1,i} + Neg_deno{1,i});
        end
    end

    Fscore_n_FeatureLabel = [F_Score; feature_label];

    A = Fscore_n_FeatureLabel';
    B = A(5:size(A,1),:);
    C = sortrows(B,1);
    Ju_Score{m} = flipud(C);
    
    clear Pos_sample;
    clear Neg_sample;
    clear Feature_sum;
    clear Feature_mean_whole;
    clear Feature_sum_pos;
    clear Feature_mean_pos;
    clear Feature_sum_neg;
    clear Feature_mean_neg;
    clear Pos_deno;
    clear Neg_deno;
    clear F_Score;
    clear A;
    clear B;
    clear C;
    
end

% Select first 300 features of both train and test sets using F_score (Ju's score)

for n = 1:6
    
    Data = CKSAAP_Train{n};
    Data_300 = Data(:,1:4);

    for i=1:300

        for j=5:size(feature_label, 2)

            if strcmp(feature_label(1,j),Ju_Score{n}(i,2)) == true

                Data_300(:,i+4) = Data(:,j);

            end
        end

    end
    Final_Train{n} = Data_300;    
    
    clear Data_300; 
    clear Data;
    
    Data = CKSAAP_Test{n};
    Data_300 = Data(:,1:4);

    for i=1:300

        for j=5:size(feature_label, 2)

            if strcmp(feature_label(1,j),Ju_Score{n}(i,2)) == true

                Data_300(:,i+4) = Data(:,j);

            end
        end

    end
    Final_Fold{n} = Data_300;
    
    clear Data_300; 
    clear Data;    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Training the Classifier and obtaining performance metrics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cd 'C:\Users\User\Desktop\libsvm-weights-3.22\matlab'

for i = 1:6
    
    clear Test_Data Train_Data test_label train_label Test_label Train_label Test Train
    % Prepare the train, test, train label and test label sets
    Test_Data = Final_Fold{i};
    Train_Data = Final_Train{i};

    for k = 1:size(Test_Data,1)
        Test(k,:) = cell2mat(Test_Data(k,5:size(Test_Data,2)));
    end
    for k = 1:size(Train_Data,1)
        Train(k,:) = cell2mat(Train_Data(k,5:size(Train_Data,2)));
    end

    test_label = cell2mat(Test_Data(:,3));
    train_label = cell2mat(Train_Data(:,3));
    Test_label = str2num(test_label);
    Train_label = str2num(train_label);
    
    model=svmtrain([],Train_label,Train,['-s 0 -t 0 -b 1']);

    [pred,acc,prob_values]=svmpredict(Test_label,Test,model,'-b 1');
    
    % Save the probability estimates and the true labels
    prediction{i} = prob_values(:,1);
    %prediction{i} = pred;
    True_label{i} = Test_label;

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
    %PredIntital(:,i) = predinitial;
    Probvalues{:,i} = prob_values;
    PredFinal(:,i) = pred;
end
Results_Avg = sum(Results,2)/6; % Average the result to get 6 fold CV
toc
% AUC calculation
classes = [True_label{1}; True_label{2}; True_label{3}; True_label{4}; True_label{5}; True_label{6}];
scores = [prediction{1}; prediction{2}; prediction{3}; prediction{4}; prediction{5}; prediction{6}];

[X4,Y4,T,AUC_4] = perfcurve(classes,scores,1);

% Display the result
metrics = {'sensitivity'; 'specificity'; 'precision'; 'accuracy'; 'mcc'};
Six_Fold_CV = [metrics, num2cell(Results_Avg)] 
AUC_4

% Plot of ROC Curve
plot(X4,Y4,'linewidth',3)
title('ROC Curve')
xlabel('False Positive Rate')
ylabel('True Positive Rate')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
