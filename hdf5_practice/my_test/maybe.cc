#include <iostream>
#include <string>
#include "H5Cpp.h"
//#include "hdf5.h"

using namespace H5;

const H5std_string FILE_NAME("testcpp.h5");
const H5std_string DATASET_NAME("eri_derivs");

const int nbf = 20;
const int deriv_stride = nbf * nbf * nbf * nbf;
const int n_partial_derivs = 4;
const int DIM = deriv_stride * n_partial_derivs;
const int RANK = 1;


//const char saveFilePath[] = "testcpp.h5";
//const std::string FileName("testcpp.h5")
//const std::string DatasetName("eri_derivs")
//const hsize_t ndim = 5;

int main(){
    // Create hdf5 file
    //hid_t file = H5Fcreate(saveFilePath, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    //H5::DataSpace space(

    H5File file(FILE_NAME, H5F_ACC_TRUNC);
    //hsize_t dims[1]; // dataset dimensions
    //dims[0] = DIM;

    //DataSpace dataspace(RANK, dims);
    //DataSet dataset = file.createDataSet(DATASET_NAME, PredType::STD_I32BE, dataspace);

    return 0;
}


