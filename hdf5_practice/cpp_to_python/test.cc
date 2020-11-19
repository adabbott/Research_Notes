#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include "H5Cpp.h"
//#include "hdf5.h"

// 
// The data space and data type are set when the dataset is created,
// and cannot be changed for the life the of dataset

using namespace H5;

const H5std_string FILE_NAME("testcpp.h5");
const H5std_string DATASET_NAME("eri_derivs");

const int nbf = 20;
const int deriv_stride = nbf * nbf * nbf * nbf;
const int n_partial_derivs = 4;
const int DIM = deriv_stride * n_partial_derivs;
const int rank = 1;


//const char saveFilePath[] = "testcpp.h5";
//const std::string FileName("testcpp.h5")
//const std::string DatasetName("eri_derivs")
//const hsize_t ndim = 5;

int main(){
    // Create hdf5 file
    //hid_t file = H5Fcreate(saveFilePath, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    //H5::DataSpace space(
    
    H5File file(FILE_NAME, H5F_ACC_TRUNC);
    hsize_t dims[1]; // dataset dimensions
    dims[0] = DIM;

    DataSpace space(rank, dims);
    // H5sstd_string name, data type, rank  
    DataSet dataset = file.createDataSet(DATASET_NAME, PredType::STD_I64BE, space);

    double eris[DIM];
    for (int i = 0; i < DIM; i++){
        eris[i] = 1.0;
    }
    // write has 5 args: 
    // buf : Buffer containing data to be written.
    // mem_type : memory data type
    // mem_space : the dataspace defining the memory dataspace
    // file_space : the dataset's dataspace on disk (file)
    // xfer_plist :: Transfer property list for this I/O operation
    dataset.write(eris, PredType::STD_I64BE);

    return 0;
}


