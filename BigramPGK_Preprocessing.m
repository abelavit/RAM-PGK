
% Algorithm below was used to extract samples from already obtained features from PSI-BLAST tool

clear all
close all

%-------------------------------------------------------------
% Data Extraction 
%-------------------------------------------------------------
tic
load Phosphoglycerylationstruct

Unprocessed_data = DB_Phosphoglycerylation;

Field = size(Unprocessed_data,2); % Columns of unprocessed data. It is num of protein sequences 

z=0;

Final_Data = cell(z,4);
k=0;  

for l = 1:Field
    z = z + size(strfind(Unprocessed_data(l).seq{1},'K'), 2); % Finding total number of -ve and +ve samples
    K_locations_field{l} = strfind(Unprocessed_data(l).seq{1},'K'); % Saves the locations of K found in protein sequences
end

window = 32; % 32 upstream and 32 downstream

for m = 1:Field
    for n=1:size(K_locations_field{m},2) % loop in all the K locations
        
        k = k+1; % Increment to next data saving location
        Final_Data{k,1} = Unprocessed_data(m).name; % Save protein name at 1st position
        
        if K_locations_field{m}(n) <= window % Location K close to N terminus
            matrix_a = Unprocessed_data(m).pssm_prob(1:K_locations_field{m}(n)+ window,:); % Matrix containing pssm_prob from start of protein seq till 32 upstream of K
            matrix_b = Unprocessed_data(m).pssm_prob(K_locations_field{m}(n)+K_locations_field{m}(n):K_locations_field{m}(n)+ window,:); % Matrix containing pssm_prob from location 2x the location of K till end of upstream 
            matrix_c = flipud(matrix_b); % Mirroring the matrix_b (flipping on horizontal axis) 
            Final_Data{k,5} = [matrix_c; matrix_a]; % Concatenate the two matrices and save at second position of the cell
       
        elseif K_locations_field{m}(n) > (Unprocessed_data(m).len - window) % Location K close to C terminus
            matrix_d = Unprocessed_data(m).pssm_prob(K_locations_field{m}(n)-window:Unprocessed_data(m).len,:); % Matrix containing pssm_prob from 32 downstream of K till end of protein seq
            matrix_e = Unprocessed_data(m).pssm_prob(K_locations_field{m}(n)-window:Unprocessed_data(m).len-(((Unprocessed_data(m).len-K_locations_field{m}(n))*2)+1),:); % Matrix containing pssm_prob downstream of K till the value to be mirrored
            matrix_f = flipud(matrix_e); % Mirroring the matrix_b (flipping on horizontal axis)
            Final_Data{k,5} = [matrix_d; matrix_f]; % Concatenate the two matrices and save at second position of the cell
            
        else % Location is good and does not require mirroring
            Final_Data{k,5} = Unprocessed_data(m).pssm_prob(K_locations_field{m}(n)-window:K_locations_field{m}(n)+window,:); % Save pssm_prob matrix (32 up and down streams of K) at second position
        
        end
        
        Final_Data{k,3} = Unprocessed_data(m).label{1}(K_locations_field{m}(n)); % Save class label at third position
        Final_Data{k,4} = K_locations_field{m}(n); % Save the K's location in the protein sequence

    end
   
end

Training_Data1 = Final_Data; 
clear Final_Data

% Bigram
for l=1:size(Training_Data1,1)
    Training_Data1{l,2} = Training_Data1{l,5}/100; % Divide pssm_prob by 100
    
    Bigram_Mat = zeros(size(Training_Data1{l,5}, 2),size(Training_Data1{l,5}, 2));
    for M=1:size(Training_Data1{l,5}, 2)
       for N=1:size(Training_Data1{l,5}, 2)
           for i=1:size(Training_Data1{l,5},1)-1
            Bigram_Mat(M,N) = Bigram_Mat(M,N) + Training_Data1{l,5}(i,M)*Training_Data1{l,5}(i+1,N); % Bigram calculation
           end
       end
    end
    transposed_Bigram_Mat = Bigram_Mat'; % Transpose Bigram_Mat
    Training_Data1{l,2} = transposed_Bigram_Mat(:)'; % Store the transposed Bigram_Mat as column vector. This gives Feature Vector
end

Final_Data = Training_Data1;
toc