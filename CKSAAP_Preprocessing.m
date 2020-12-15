
clear all
close all

tic
load Phosphoglycerylationstruct % Load the phosphoglycerylation dataset (raw file from which we will extract the data from)

Unprocessed_data = DB_Phosphoglycerylation;

Field = size(Unprocessed_data,2); % Columns of unprocessed data. It is num of protein sequences 

k=0; 
z=0;
for a = 1:Field
    z = z + size(strfind(Unprocessed_data(a).seq{1},'K'), 2); % Finding total number of -ve and +ve samples
    K_locations_field{a} = strfind(Unprocessed_data(a).seq{1},'K'); % Saves the locations of K found in protein sequences
end
Final_Data = cell(z,4);

X_sequences = {'X' 'XX' 'XXX' 'XXXX' 'XXXXX' 'XXXXXX' 'XXXXXXX'};

for m = 1:Field
    for n=1:size(K_locations_field{m},2) % loop in all the K locations
        
        k = k+1; % Increment to next data saving location
        Final_Data{k,1} = Unprocessed_data(m).name; % Save protein name at 1st position
        
        if K_locations_field{m}(n) < 8 % Location K close to N terminus
            sequence_a = Unprocessed_data(m).seq{1}(1:K_locations_field{m}(n)+7); % Sequence containing the start of protein seq till 7 upstream of K
            sequence_b = X_sequences{8-K_locations_field{m}(n)}; % Find the number of 'X' in X_sequences to match with those missing amino acids
 
            Final_Data{k,2} = [sequence_b sequence_a]; % Concatenate sequence b and a and save at second position of the cell
       
        elseif K_locations_field{m}(n) > (Unprocessed_data(m).len - 7) % Location K close to C terminus
            sequence_a = Unprocessed_data(m).seq{1}(K_locations_field{m}(n)-7:Unprocessed_data(m).len); % Sequence containing 7 downstream of K till end of protein seq
            sequence_b = X_sequences{7-(Unprocessed_data(m).len - K_locations_field{m}(n))}; % Find the number of 'X' in X_sequences to match with those missing amino acids
            
            Final_Data{k,2} = [sequence_a sequence_b]; % Concatenate sequence b and a and save at second position of the cell
       
        else % Location is good and does not require mirroring
            Final_Data{k,2} = Unprocessed_data(m).seq{1}(K_locations_field{m}(n)-7:K_locations_field{m}(n)+7); % Save proetin sequence (15 up and down stream of K) at second position
        
        end
        
        Final_Data{k,3} = Unprocessed_data(m).label{1}(K_locations_field{m}(n)); % Save class label at third position
        Final_Data{k,4} = K_locations_field{m}(n); % Save the K's location in the protein sequence
    
    end
end





Amino_acids = {'A' 'C' 'D' 'E' 'F' 'G' 'H' 'I' 'K' 'L' 'M' 'N' 'P' 'Q' 'R' 'S' 'T' 'V' 'W' 'X' 'Y'};

% k = 0
for h = 1:size(Final_Data, 1)
    sequence = Final_Data{h,2};
    feature_num = 0;
    for i = 1:size(Amino_acids, 2)
        for j = 1:size(Amino_acids, 2)
            feature_num = feature_num + 1;
            s = strcat(Amino_acids{i}, Amino_acids{j}); % concatenate 2 amino acids 
            match = strfind(sequence, s); % find the concatenated amino acids in the sequence/peptide
            Num = size(match, 2); % store the number of matches
            Final_Data{h,feature_num + 4} = Num; % save the number as its feature
            feature_label {1,feature_num + 4} = s;
        end
    end
end


% k = 1
for h = 1:size(Final_Data, 1)
    sequence = Final_Data{h,2};
    feature_num = 0;
    for i = 1:size(Amino_acids, 2)
        for j = 1:size(Amino_acids, 2)
            feature_num = feature_num + 1;
            s = strcat(Amino_acids{i}, '(?=.', Amino_acids{j},')'); % concatenate 2 amino acids. Dot is added to indicate there can be any amino acid in between (concatenate to come to this form for e.g. X(?=.X))
            match = regexp(sequence, s); % find the concatenated amino acids in the sequence/peptide
            Num = size(match, 2); % store the number of matches
            Final_Data{h,feature_num + 445} = Num; % save the number as its feature
            feature_label {1,feature_num + 445} = s;
        end
    end
end

% k = 2
for h = 1:size(Final_Data, 1)
    sequence = Final_Data{h,2};
    feature_num = 0;
    for i = 1:size(Amino_acids, 2)
        for j = 1:size(Amino_acids, 2)
            feature_num = feature_num + 1;
            s = strcat(Amino_acids{i}, '(?=..', Amino_acids{j},')'); % concatenate 2 amino acids. Dot is added to indicate there can be any amino acid in between (concatenate to come to this form for e.g. X(?=..X))
            match = regexp(sequence, s); % find the concatenated amino acids in the sequence/peptide
            Num = size(match, 2); % store the number of matches
            Final_Data{h,feature_num + 886} = Num; % save the number as its feature
            feature_label {1,feature_num + 886} = s;
        end
    end
end

% k = 3
for h = 1:size(Final_Data, 1)
    sequence = Final_Data{h,2};
    feature_num = 0;
    for i = 1:size(Amino_acids, 2)
        for j = 1:size(Amino_acids, 2)
            feature_num = feature_num + 1;
            s = strcat(Amino_acids{i}, '(?=...', Amino_acids{j},')'); % concatenate 2 amino acids. Dot is added to indicate there can be any amino acid in between (concatenate to come to this form for e.g. X(?=...X))
            match = regexp(sequence, s); % find the concatenated amino acids in the sequence/peptide
            Num = size(match, 2); % store the number of matches
            Final_Data{h,feature_num + 1327} = Num; % save the number as its feature
            feature_label {1,feature_num + 1327} = s;
        end
    end
end

% k = 4
for h = 1:size(Final_Data, 1)
    sequence = Final_Data{h,2};
    feature_num = 0;
    for i = 1:size(Amino_acids, 2)
        for j = 1:size(Amino_acids, 2)
            feature_num = feature_num + 1;
            s = strcat(Amino_acids{i}, '(?=....', Amino_acids{j},')'); % concatenate 2 amino acids. Dot is added to indicate there can be any amino acid in between (concatenate to come to this form for e.g. X(?=....X))
            match = regexp(sequence, s); % find the concatenated amino acids in the sequence/peptide
            Num = size(match, 2); % store the number of matches
            Final_Data{h,feature_num + 1768} = Num; % save the number as its feature
            feature_label {1,feature_num + 1768} = s;
        end
    end
end
CKSAAP_Data = Final_Data;
toc
            