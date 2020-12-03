
function [BigramPGK_Set] = Get_BigramPGK(Set, BigramPGK_Data)

for i=1:size(Set,1)

        for j=1:size(BigramPGK_Data,1)

            if (strcmp(Set{i,1}, BigramPGK_Data{j,1}) == true ) && (Set{i,4} == BigramPGK_Data{j,4})

                       BigramPGK_Set(i,:) = BigramPGK_Data(j,:);

            end
        end
end