clear all
close all
%-------------------------------------------------------------
% Data Extraction 
%-------------------------------------------------------------
tic
load Phosphoglycerylationstruct % Load the phosphoglycerylation dataset (raw file from which we will extract the data from)

% Data extraction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Unprocessed_data = DB_Phosphoglycerylation;

Field = size(Unprocessed_data,2); % Columns of unprocessed data. It is number of protein sequences 

z=0;
k=0; 
for a = 1:Field
    z = z + size(strfind(Unprocessed_data(a).seq{1},'K'), 2); % Finding total lysine K in the entire protein sequences
    AA_locations_seq{1,a} = strfind(Unprocessed_data(a).seq{1},'A'); % Saves the locations of A found in protein sequences
    AA_locations_seq{2,a} = strfind(Unprocessed_data(a).seq{1},'R'); % Saves the locations of R found in protein sequences
    AA_locations_seq{3,a} = strfind(Unprocessed_data(a).seq{1},'N'); % Saves the locations of N found in protein sequences
    AA_locations_seq{4,a} = strfind(Unprocessed_data(a).seq{1},'D'); % Saves the locations of D found in protein sequences
    AA_locations_seq{5,a} = strfind(Unprocessed_data(a).seq{1},'C'); % Saves the locations of C found in protein sequences
    AA_locations_seq{6,a} = strfind(Unprocessed_data(a).seq{1},'Q'); % Saves the locations of Q found in protein sequences
    AA_locations_seq{7,a} = strfind(Unprocessed_data(a).seq{1},'E'); % Saves the locations of E found in protein sequences
    AA_locations_seq{8,a} = strfind(Unprocessed_data(a).seq{1},'G'); % Saves the locations of G found in protein sequences
    AA_locations_seq{9,a} = strfind(Unprocessed_data(a).seq{1},'H'); % Saves the locations of H found in protein sequences
    AA_locations_seq{10,a} = strfind(Unprocessed_data(a).seq{1},'I'); % Saves the locations of I found in protein sequences
    AA_locations_seq{11,a} = strfind(Unprocessed_data(a).seq{1},'L'); % Saves the locations of L found in protein sequences
    
    AA_locations_seq{12,a} = strfind(Unprocessed_data(a).seq{1},'K'); % Saves the locations of K found in protein sequences

    AA_locations_seq{13,a} = strfind(Unprocessed_data(a).seq{1},'M'); % Saves the locations of M found in protein sequences
    AA_locations_seq{14,a} = strfind(Unprocessed_data(a).seq{1},'F'); % Saves the locations of F found in protein sequences
    AA_locations_seq{15,a} = strfind(Unprocessed_data(a).seq{1},'P'); % Saves the locations of P found in protein sequences
    AA_locations_seq{16,a} = strfind(Unprocessed_data(a).seq{1},'S'); % Saves the locations of S found in protein sequences
    AA_locations_seq{17,a} = strfind(Unprocessed_data(a).seq{1},'T'); % Saves the locations of T found in protein sequences
    AA_locations_seq{18,a} = strfind(Unprocessed_data(a).seq{1},'W'); % Saves the locations of W found in protein sequences
    AA_locations_seq{19,a} = strfind(Unprocessed_data(a).seq{1},'Y'); % Saves the locations of Y found in protein sequences
    AA_locations_seq{20,a} = strfind(Unprocessed_data(a).seq{1},'V'); % Saves the locations of V found in protein sequences

end

n_AA = 6; % define n

for m = 1:Field
    for n=1:size(AA_locations_seq{12,m},2) % loop in all the K locations
        
        k = k+1; % Increment to next data saving location
        Final_Data{k,1} = Unprocessed_data(m).name; % Save protein name at 1st position
        Final_Data{k,3} = Unprocessed_data(m).label{1}(AA_locations_seq{12,m}(n)); % Save class label
        Final_Data{k,4} = AA_locations_seq{12,m}(n); % Save the K's location in the protein sequence
        Final_Data{k,6} = m;
        
        % First get all AA_Distance
        for o = 1:20
            AA_distance{o,1} = sort(abs(AA_locations_seq{o,m} - AA_locations_seq{12,m}(n))); % Loop through all 20 amino acid locations in the same seq and find absolute distance and sort them
            
            if size(AA_distance{o,1},2) > n_AA
                
                AA_distance{o,1} = AA_distance{o,1}(1:n_AA); % If distances is greater than n_AA, clip off the extra
                
            end
        end
        
        % Now construct feature
        for o=1:20 
            
            if size(AA_distance{o,1}) == 0 % If there were no locations for the amino acid
           
                % calculate overall average to fill for the missing amino acid
                ForMissingAA = AA_distance;
                for AAinAA_Distance = 1:20 
                    
                    ForMissingAA{AAinAA_Distance,2} = sum(ForMissingAA{AAinAA_Distance,1});
                    ForMissingAA{AAinAA_Distance,3} = size(ForMissingAA{AAinAA_Distance,1},2);
                    
                end
                distance_vec_all_AA = ForMissingAA(:,2);
                count_vec_all_AA = ForMissingAA(:,3);
                RAM_feat_for_missing_AA = sum(cell2mat(distance_vec_all_AA))/sum(cell2mat(count_vec_all_AA));
                
                b = RAM_feat_for_missing_AA; % store overall mean of amino acids when a particular amino acid is not present in the sequence
                
            else  
                    
                b = AA_distance{o,1}; 
                
                if size(AA_distance{o,1}, 2) < n_AA % if number of that particular amino acid is less than n_AA
                    
                    c = AA_distance{o,1};
                    
                    additional_dist = n_AA - size(AA_distance{o,1},2);
                    
                    for p = 1:additional_dist
                        
                        d(p) = sum(AA_distance{o,1})/size(AA_distance{o,1}, 2);
                        
                    end
                    b = [c d];
                    
                end
                
            end
            AA_Min_Dists(o,:) = b;
            clear b
            clear c
            clear d
            
        end
        Final_Data{k,5} = AA_Min_Dists; % Store the min distances of the 20 amino acid to the lysine
        Final_Data{k,2} = transpose(Final_Data{k,5}(:)); % rearrange to get vector 
        vec = cell2mat(Final_Data(k,2));
        [M,I] = max(vec);
        Final_Data{k,2} = Final_Data{k,2}/M;
    end
end
toc
