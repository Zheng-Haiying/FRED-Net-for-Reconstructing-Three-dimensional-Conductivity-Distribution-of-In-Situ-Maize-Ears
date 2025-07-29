# FRED-Net-for-Reconstructing-Three-dimensional-Conductivity-Distribution-of-In-Situ-Maize-Ears
This is the open-source code for a research paper titled “Reconstructing Three-dimensional Conductivity Distribution of In-Situ Maize Ears Using Frequency-Enhanced Residual Encoder-Decoder Network.” I will provide full reproducibility after the paper is accepted.

The article has now been published, the link is https://doi.org/10.1016/j.engappai.2025.111858

The networks folder contains multiple comparison models including ResUNet. 
——1.Before network training, you need to generate appropriate training data using "create_maize_dataset_for_train.m" and "create_threeobj_dataset_for_train.m". 
——2."main.py" is the network training code, which is the most important file. 
——3.After model training, you can compare training results using "model_evaluation_gui.py". 
——4.The two "visualizate.m" functions are for visualizing and imaging the generated conductivity data. 
——5."requirements.txt" contains the required packages.

If you have any questions about the code, please contact us promptly. Due to the large size of the datasets, we have not uploaded them to the repository, please understand. You can create the required datasets based on EIDORS“https://eidors3d.sourceforge.net/tutorial/tutorial.shtml” using the dataset generation code.
