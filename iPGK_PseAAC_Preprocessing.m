

clear all
close all
tic
load Phosphoglycerylationstruct % Load the phosphoglycerylation dataset (raw file from which we will extract the data from)

Unprocessed_data = DB_Phosphoglycerylation;

Field = size(Unprocessed_data,2); % Columns of unprocessed data. It is num of protein sequences 

z=0;

iPGK_Data = cell(z,4);
k=0;  

for l = 1:Field
    z = z + size(strfind(Unprocessed_data(l).seq{1},'K'), 2); % Finding total number of -ve and +ve samples
    K_locations_field{l} = strfind(Unprocessed_data(l).seq{1},'K'); % Saves the locations of K found in protein sequences
end

X_sequences = {'X' 'XX' 'XXX' 'XXXX' 'XXXXX' 'XXXXXX' 'XXXXXXX'};

for m = 1:Field
    for n=1:size(K_locations_field{m},2) % loop in all the K locations
        
        k = k+1; % Increment to next data saving location
        iPGK_Data{k,1} = Unprocessed_data(m).name; % Save protein name at 1st position
        
        if K_locations_field{m}(n) < 8 % Location K close to N terminus
            sequence_a = Unprocessed_data(m).seq{1}(1:K_locations_field{m}(n)+7); % Sequence containing the start of protein seq till 7 upstream of K
            sequence_b = X_sequences{8-K_locations_field{m}(n)}; % Find the number of 'X' in X_sequences to match with those missing amino acids
 
            iPGK_Data{k,5} = [sequence_b sequence_a]; % Concatenate sequence b and a and save at second position of the cell
       
        elseif K_locations_field{m}(n) > (Unprocessed_data(m).len - 7) % Location K close to C terminus
            sequence_a = Unprocessed_data(m).seq{1}(K_locations_field{m}(n)-7:Unprocessed_data(m).len); % Sequence containing 7 downstream of K till end of protein seq
            sequence_b = X_sequences{7-(Unprocessed_data(m).len - K_locations_field{m}(n))}; % Find the number of 'X' in X_sequences to match with those missing amino acids
            
            iPGK_Data{k,5} = [sequence_a sequence_b]; % Concatenate sequence b and a and save at second position of the cell
       
        else % Location is good and does not require mirroring
            iPGK_Data{k,5} = Unprocessed_data(m).seq{1}(K_locations_field{m}(n)-7:K_locations_field{m}(n)+7); % Save proetin sequence (15 up and down stream of K) at second position
        
        end
        
        iPGK_Data{k,3} = Unprocessed_data(m).label{1}(K_locations_field{m}(n)); % Save class label at third position
        iPGK_Data{k,4} = K_locations_field{m}(n); % Save the K's location in the protein sequence
    
    end
end

for i = 1:size(iPGK_Data,1)

    Amino_acids = cellstr(iPGK_Data{i,5}(:))';
    
    % First tier
    for k = 1:14
        t1{1,k} = strcat(Amino_acids{k}, Amino_acids{k+1});    
    end
    
    for k = 1:14
        count = 0;
        for l = 1:14
            if (t1{1,k} == t1{1,l})
                count = count + 1;
            end
        end
        t1{2,k} = count;
    end
    
    % second tier
    for k = 1:13  
        t2{1,k} = strcat(Amino_acids{k}, '.', Amino_acids{k+2}); 
    end
    for k = 1:13
        count = 0;
        for l = 1:13
            if (t2{1,k} == t2{1,l})
                count = count + 1;
            end
        end
        t2{2,k} = count;
    end
    
    % Third tier
    for k = 1:12
        t3{1,k} = strcat(Amino_acids{k}, '..', Amino_acids{k+3}); 
    end
    for k = 1:12
        count = 0;
        for l = 1:12
            if (t3{1,k} == t3{1,l})
                count = count + 1;
            end
        end
        t3{2,k} = count;
    end
    
    % Fourth tier
    for k = 1:11 
        t4{1,k} = strcat(Amino_acids{k}, '...', Amino_acids{k+4}); 
    end
    for k = 1:11
        count = 0;
        for l = 1:11
            if (t4{1,k} == t4{1,l})
                count = count + 1;
            end
        end
        t4{2,k} = count;
    end
    iPGK_Data{i,2} = cell2mat({t1{2,:} t2{2,:} t3{2,:} t4{2,:}});
end
toc