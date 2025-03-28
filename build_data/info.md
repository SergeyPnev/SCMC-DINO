
## Datasets

The datasets can be downloaded from their respective sources: 

[RxRx1](https://www.rxrx.ai/rxrx1)

[CPG0004](https://github.com/broadinstitute/cellpainting-gallery)

The datasets used in this work were downloaded from the above links.

Formating and preparing the datasets can be done using the code in this folder. 

For RXRX1 the data can be made into a h5py file using "build_rxrx1_h5py_file.py".

For CPG0004 the data can be downloaded from the Cell Painting Gallery in the above link. 
The data is downloaded using AWS CLI. The preprocessed png files were downloaded along with the metadata found in the associated folder.

Once downloaded the the dataset was prepared using the CPG0004-build-combined.ipynb and downsize_imgs.py.

