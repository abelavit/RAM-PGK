
function [iPGK_Set] = Get_iPGK(Set, iPGK_Data)

for i=1:size(Set,1)

        for j=1:size(iPGK_Data,1)

            if (strcmp(Set{i,1}, iPGK_Data{j,1}) == true ) && (Set{i,4} == iPGK_Data{j,4})

                       iPGK_Set(i,:) = iPGK_Data(j,:);

            end
        end
end