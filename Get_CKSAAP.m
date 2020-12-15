
function [CKSAAP_Set] = Get_CKSAAP(Set, CKSAAP_Data)

for i=1:size(Set,1)

        for j=1:size(CKSAAP_Data,1)

            if (strcmp(Set{i,1}, CKSAAP_Data{j,1}) == true ) && (Set{i,4} == CKSAAP_Data{j,4})

                       CKSAAP_Set(i,:) = CKSAAP_Data(j,:);

            end
        end
end