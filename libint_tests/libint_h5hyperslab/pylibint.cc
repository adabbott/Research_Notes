#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdlib.h>
#include <libint2.hpp>
#include <H5Cpp.h>

namespace py = pybind11;
using namespace H5;

std::vector<libint2::Atom> atoms;
libint2::BasisSet obs;
int nbf;
int natom;
int ncart;
std::vector<size_t> shell2bf;
std::vector<long> shell2atom;

std::vector<libint2::Atom> get_atoms(std::string xyzfilename)
{
    std::ifstream input_file(xyzfilename);
    std::vector<libint2::Atom> atoms = libint2::read_dotxyz(input_file);
    return atoms;
}

void initialize(std::string xyzfilename, std::string basis_name) {
    libint2::initialize();
    atoms = get_atoms(xyzfilename);
    // Move harddrive load of basis and xyz to happen only once
    obs = libint2::BasisSet(basis_name, atoms);
    obs.set_pure(false); // use cartesian gaussians
    // Get size of potential derivative array and allocate 
    nbf = obs.nbf();
    natom = atoms.size();
    ncart = natom * 3;
    shell2bf = obs.shell2bf(); // maps shell index to basis function index
    shell2atom = obs.shell2atom(atoms); // maps shell index to atom index
}

void finalize() {
    libint2::finalize();
}

py::array overlap() {
    // Overlap integral engine
    libint2::Engine s_engine(libint2::Operator::overlap,obs.max_nprim(),obs.max_l());
    const auto& buf_vec = s_engine.results(); // will point to computed shell sets
    size_t length = nbf * nbf;
    std::vector<double> result(length); // vector to store integral array

    for(auto s1=0; s1!=obs.size(); ++s1) {
        auto bf1 = shell2bf[s1];  // first basis function in first shell
        auto n1 = obs[s1].size(); // number of basis functions in first shell
        for(auto s2=0; s2!=obs.size(); ++s2) {
            auto bf2 = shell2bf[s2];  // first basis function in second shell
            auto n2 = obs[s2].size(); // number of basis functions in second shell

            s_engine.compute(obs[s1], obs[s2]); // Compute shell set
            auto ints_shellset = buf_vec[0];    // Location of the computed integrals
            if (ints_shellset == nullptr)
                continue;  // nullptr returned if the entire shell-set was screened out
            // Loop over shell block, keeping a total count idx for the size of shell set
            for(auto f1=0, idx=0; f1!=n1; ++f1) {
                for(auto f2=0; f2!=n2; ++f2, ++idx) {
                    result[(bf1 + f1) * nbf + bf2 + f2] = ints_shellset[idx];
                }
            }
        }
    }
    return py::array(result.size(), result.data());
}

void overlap_disk() {
    // Overlap integral engine
    libint2::Engine s_engine(libint2::Operator::overlap,obs.max_nprim(),obs.max_l());
    const auto& buf_vec = s_engine.results(); // will point to computed shell sets
    size_t length = nbf * nbf;
    //std::vector<double> result(length); // vector to store integral array

    // Begin HDF5 jargon.
    // Define file dataspace dimensions
    const int fspace_dim1 = nbf;
    const int fspace_dim2 = nbf;
    // Define file dataspace rank
    const int fspace_rank = 2;
    // Define file name
    const H5std_string file_name("overlap.h5");
    // Define dataset name
    const H5std_string dataset_name("overlap");
    // Create H5 File and fill with 0.0's
    H5File* file = new H5File(file_name,H5F_ACC_TRUNC);
    double fillvalue = 0.0;
    DSetCreatPropList plist;
    plist.setFillValue(PredType::NATIVE_DOUBLE, &fillvalue);
    // Define dimensions of file array
    hsize_t file_dims[] = {fspace_dim1, fspace_dim2};
    // Create dataspace for file array
    DataSpace fspace(fspace_rank, file_dims);
    // Create dataset and write 0.0's into the file 
    DataSet* dataset = new DataSet(file->createDataSet(dataset_name, PredType::NATIVE_DOUBLE, fspace, plist));
    // Now in the loops, define hyperslab start, stride, count and sizes
    // End HDF5 jargon.

    for(auto s1=0; s1!=obs.size(); ++s1) {
        auto bf1 = shell2bf[s1];  // first basis function in first shell
        auto n1 = obs[s1].size(); // number of basis functions in first shell
        for(auto s2=0; s2!=obs.size(); ++s2) {
            auto bf2 = shell2bf[s2];  // first basis function in second shell
            auto n2 = obs[s2].size(); // number of basis functions in second shell

            s_engine.compute(obs[s1], obs[s2]); // Compute shell set

            // Define shell set slab
            double shellset_slab [n1][n2];

            auto ints_shellset = buf_vec[0];    // Location of the computed integrals
            if (ints_shellset == nullptr)
                continue;  // nullptr returned if the entire shell-set was screened out
            // Loop over shell block, keeping a total count idx for the size of shell set
            for(auto f1=0, idx=0; f1!=n1; ++f1) {
                for(auto f2=0; f2!=n2; ++f2, ++idx) {
                    //result[(bf1 + f1) * nbf + bf2 + f2] = ints_shellset[idx];
                    shellset_slab[f1][f2] = ints_shellset[idx];
                }
            }
            // TODO Now write this shell set slab to HDF5 file
            // We need to define fspace slab, where data will be dumped, 
            //    and mspace slab, where we will obtain the data from,
            // and then write it to disk
            hsize_t start[2];
            hsize_t stride[2];
            hsize_t count[2];
            hsize_t block[2];
            start[0] = bf1;
            start[1] = bf2;
            count[0]  = n1; 
            count[1]  = n2;
            stride[0] = 1;
            stride[1] = 1;
            block[0]  = 1; block[1]  = 1;
            fspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);
            // Create dataspace for memory dataset to write to file
            // Hmmm anything that is 'const' here is a problem right?
            hsize_t mem_dims[] = {n1, n2};
            DataSpace mspace(2, mem_dims);
            start[0] = 0;
            start[1] = 0;
            mspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);
            //DataSpace memspace(mspace_rank,  );
            // write buffer data 'shellset_slab' with data type double from memory dataspace mspace to file dataspace fspace 
            dataset->write(shellset_slab, PredType::NATIVE_DOUBLE, mspace, fspace);
        }
    }
    // closes the dataset and file
    delete dataset;
    delete file;

    //const int dim1 = length; 
    //const int rank = 1;

    // Array of dimensions, since just one dimensional have just one element with size = length = total overlap integrals
    //hsize_t dimsf[1];
    //dimsf[0] = dim1;

    // Write the array to HDF5 File.
    // Second argument has the following options:
    // H5F_ACC_TRUNC - Truncate file, if it already exists, erasing all data previously stored in the file.
    // H5F_ACC_EXCL - Fail if file already exists. H5F_ACC_TRUNC and H5F_ACC_EXCL are mutually exclusive
    // H5F_ACC_RDONLY - Open file as read-only, if it already exists, and fail, otherwise
    // H5F_ACC_RDWR - Open file for read/write, if it already exists, and fail, otherwise

    //H5File file(file_name, H5F_ACC_TRUNC);
    //DataSpace space(rank, dimsf);
    //DataSet dataset = file.createDataSet(dataset_name, PredType::NATIVE_DOUBLE, space);
    //dataset.write(result, PredType::NATIVE_DOUBLE);
}

PYBIND11_MODULE(pylibint, m) {
    m.def("initialize", &initialize, "Initializes libint, builds geom and basis, assigns globals");
    m.def("finalize", &finalize, "Kills libint");
    m.def("overlap", &overlap, "Computes overlap integrals with libint and return NumPy array"); 
    m.def("overlap_disk", &overlap_disk, "Computes overlap integrals and writes them to disk with HDF5");
}


