# Cancer-prediction-using-SVM
This dataset had variables which shows the dimensions and nature of a tumour and we have to predict whether 
it will be beniegn or malignant based on the experience of previous values
so we divide the dataset in test and training to train the data and validate on test set.
following are the variables
"id"                       "diagnosis"               "radius_mean"            
"texture_mean"             "perimeter_mean"          "area_mean"              
 "smoothness_mean"          "compactness_mean"        "concavity_mean"         
"concave.points_mean"      "symmetry_mean"           "fractal_dimension_mean" 
"radius_se"                "texture_se"              "perimeter_se"           
"area_se"                  "smoothness_se"           "compactness_se"         
"concavity_se"             "concave.points_se"       "symmetry_se"            
"fractal_dimension_se"     "radius_worst"            "texture_worst"          
"perimeter_worst"          "area_worst"              "smoothness_worst"       
"compactness_worst"        "concavity_worst"         "concave.points_worst"   
"symmetry_worst"           "fractal_dimension_worst" "X" 
We use kernel SVM classifier to classify the variables.
We have used PCA to extract 2 highly correlated features and visualise dataset
To imporve the accuracy the PCA method should NOT BE USED for this dataset.
Only for having a better visualisation we have used the PCA.
The Visualisation result with PCA being used is attached in the repository.
