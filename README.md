# RAM-PGK

1. To obtain 6-Fold Cross-Validation results, please run:
- 'iPGK_PseAAC_Method' for iPGK-PseAAC result
- 'BigramPGK_Method' for Bigram-PGK result
- 'RAM_PGK_Method' for RAM-PGK result (Feature Selected via backward elimination)
- 'RAM_PGK_Method_NoFeatureSelection' for RAM-PGK (without feature selection) result 

The above scripts calculate the 6-fold cross-validation on the same datasets. This is found in the file named 'ModelData'. 

2. To obtain feature construction of the methods, please run:
- 'iPGK_PseAAC_Preprocessing' for iPGK-PseAAC features
- 'BigramPGK_Preprocessing' for Bigram-PGK features
- 'RAM_PGK_Preprocessing' for RAM-PGK features

'Phosphoglycerylationstruct' file contains the protein sequences with true labels used in this work. 
