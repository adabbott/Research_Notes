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
unsigned int nbf;
unsigned int natom;
unsigned int ncart;
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

    // Begin HDF5 jargon. Define file name and dataset name within file
    const H5std_string file_name("overlap.h5");
    const H5std_string dataset_name("overlap");
    // Create H5 File and prepare to fill with 0.0's
    H5File* file = new H5File(file_name,H5F_ACC_TRUNC);
    double fillvalue = 0.0;
    DSetCreatPropList plist;
    plist.setFillValue(PredType::NATIVE_DOUBLE, &fillvalue);
    // Create dataspace for file array (rank, dimensions)
    hsize_t file_dims[] = {nbf, nbf};
    DataSpace fspace(2, file_dims);
    // Create dataset and write 0.0's into the file 
    DataSet* dataset = new DataSet(file->createDataSet(dataset_name, PredType::NATIVE_DOUBLE, fspace, plist));
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
            // Now write this shell set slab to HDF5 file
            // Create file space hyperslab, defining where to write data to in file
            hsize_t count[2] = {n1, n2};
            hsize_t start[2] = {bf1, bf2};
            hsize_t stride[2] = {1,1}; // stride and block
            hsize_t block[2] = {1,1};  // can be used to add value to multiple places
            fspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);
            // Create dataspace defining for memory dataset to write to file
            hsize_t mem_dims[] = {n1, n2};
            DataSpace mspace(2, mem_dims);
            start[0] = 0;
            start[1] = 0;
            mspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);
            // Write buffer data 'shellset_slab' with data type double from memory dataspace `mspace` to file dataspace `fspace`
            dataset->write(shellset_slab, PredType::NATIVE_DOUBLE, mspace, fspace);
        }
    }
    // closes the dataset and file
    delete dataset;
    delete file;
}


// Writes ERI's to disk 
void eri_disk() { 
    libint2::Engine eri_engine(libint2::Operator::coulomb,obs.max_nprim(),obs.max_l());
    const auto& buf_vec = eri_engine.results(); // will point to computed shell sets
    size_t length = nbf * nbf * nbf * nbf;
    //std::vector<double> result(length);

    // Begin HDF5 jargon. Define file name and dataset name within file
    const H5std_string file_name("eri.h5");
    const H5std_string dataset_name("eri");
    // Create H5 File and prepare to fill with 0.0's
    H5File* file = new H5File(file_name,H5F_ACC_TRUNC);
    double fillvalue = 0.0;
    DSetCreatPropList plist;
    plist.setFillValue(PredType::NATIVE_DOUBLE, &fillvalue);
    // Create dataspace for file array (rank, dimensions)
    hsize_t file_dims[] = {nbf, nbf, nbf, nbf};
    DataSpace fspace(4, file_dims);
    // Create dataset and write 0.0's into the file 
    DataSet* dataset = new DataSet(file->createDataSet(dataset_name, PredType::NATIVE_DOUBLE, fspace, plist));
    // End HDF5 jargon.

    for(auto s1=0; s1!=obs.size(); ++s1) {
        auto bf1 = shell2bf[s1];  // first basis function in first shell
        auto n1 = obs[s1].size(); // number of basis functions in first shell
        for(auto s2=0; s2!=obs.size(); ++s2) {
            auto bf2 = shell2bf[s2];  // first basis function in second shell
            auto n2 = obs[s2].size(); // number of basis functions in second shell
            for(auto s3=0; s3!=obs.size(); ++s3) {
                auto bf3 = shell2bf[s3];  // first basis function in third shell
                auto n3 = obs[s3].size(); // number of basis functions in third shell
                for(auto s4=0; s4!=obs.size(); ++s4) {
                    auto bf4 = shell2bf[s4];  // first basis function in fourth shell
                    auto n4 = obs[s4].size(); // number of basis functions in fourth shell

                    eri_engine.compute(obs[s1], obs[s2], obs[s3], obs[s4]); // Compute shell set

                    // Define shell set slab
                    double shellset_slab [n1][n2][n3][n4];

                    auto ints_shellset = buf_vec[0];    // Location of the computed integrals
                    if (ints_shellset == nullptr)
                        continue;  // nullptr returned if the entire shell-set was screened out
                    // Loop over shell block, keeping a total count idx for the size of shell set
                    for(auto f1=0, idx=0; f1!=n1; ++f1) {
                        for(auto f2=0; f2!=n2; ++f2) {
                            for(auto f3=0; f3!=n3; ++f3) {
                                for(auto f4=0; f4!=n4; ++f4, ++idx) {
                                    shellset_slab[f1][f2][f3][f4] = ints_shellset[idx];
                                }
                            }
                        }
                    }
                    // Now write this shell set slab to HDF5 file
                    hsize_t count[4] = {n1, n2, n3, n4};
                    hsize_t start[4] = {bf1, bf2, bf3, bf4};
                    hsize_t stride[4] = {1,1,1,1}; // stride and block can be used to 
                    hsize_t block[4] = {1,1,1,1};  // add values to multiple places
                    fspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);
                    // Create dataspace defining for memory dataset to write to file
                    hsize_t mem_dims[] = {n1, n2, n3, n4};
                    DataSpace mspace(4, mem_dims);
                    start[0] = 0;
                    start[1] = 0;
                    start[2] = 0;
                    start[3] = 0;
                    mspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);
                    // Write buffer data 'shellset_slab' with data type double from memory dataspace `mspace` to file dataspace `fspace`
                    dataset->write(shellset_slab, PredType::NATIVE_DOUBLE, mspace, fspace);
                }
            }
        }
    }
    // closes the dataset and file
    delete dataset;
    delete file;
}

// how many unique nth order derivatives wrt k objects which have 3 differentiable coordinates each
int how_many_derivs(int k, int n) {
    int val = 1;
    int factorial = 1;
    for (int i=0; i < n; i++) {
        val *= (3 * k + i);
        factorial *= i + 1;
    }
    val /= factorial;
    return val;
}

void cwr_recursion(std::vector<int> inp,
                   std::vector<int> &out,
                   std::vector<std::vector<int>> &result,
                   int k, int i, int n)
{
    // base case: if combination size is k, add to result 
    if (out.size() == k){
        result.push_back(out);
        return;
    }
    for (int j = i; j < n; j++){
        out.push_back(inp[j]);
        cwr_recursion(inp, out, result, k, j, n);
        // backtrack - remove current element from solution
        out.pop_back();
    }
}

std::vector<std::vector<int>> generate_multi_index_lookup(int nparams, int deriv_order) {
    using namespace std;
    // Generate vector of indices 0 through nparams-1
    vector<int> inp;
    for (int i = 0; i < nparams; i++) {
        inp.push_back(i);
    }
    // Generate all possible combinations with repitition. 
    // These are upper triangle indices, and the length of them is the total number of derivatives
    vector<int> out;
    vector<vector<int>> combos;
    cwr_recursion(inp, out, combos, deriv_order, 0, nparams);
    return combos;
}

//TODO eventually use this, but needs to be optimized heavily. WIP
//std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>> get_index_map (int deriv_order, int ncenter) {
//    using namespace std;
//    int nshell_derivs = how_many_derivs(ncenter, deriv_order);
//    // Create mapping from 1d buffer index (flattened upper triangle shell derivative index) to multidimensional index
//    const std::vector<std::vector<int>> buffer_multidim_lookup = generate_multi_index_lookup(ncenter * 3, deriv_order);
//    // Create mapping from 1d cartesian coodinate index (flattened upper triangle cartesian derivative index) to multidimensional index
//    const std::vector<std::vector<int>> cart_multidim_lookup = generate_multi_index_lookup(natom * 3, deriv_order);
//    std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>> mapping(nshell_derivs, vector<vector<vector<vector<int>>>> 
//                                                                                (natom, vector<vector<vector<int>>>
//                                                                                (natom, vector<vector<int>>
//                                                                                (natom, vector<int>
//                                                                                (natom, 0)))));
//
//    for (int z = 0; z < nshell_derivs; z++) {
//        auto multi_shell_idx = buffer_multidim_lookup[z];
//        // Save center index and component index for each multidimensional shell index.
//        std::vector<std::vector<int>> tmp_indices;
//        for (auto shell_idx : multi_shell_idx){
//          div_t tmp = std::div(shell_idx, 3);
//          //int center_idx = tmp.quot;
//          //int comp_idx = tmp.rem;
//          //int atom_idx = shell_atom_index_list[center_idx];
//          //multi_cart_idx.push_back(3 * atom_idx + comp_idx);
//          std::vector<int> tmp_vec = {tmp.quot, tmp.rem};
//          tmp_indices.push_back(tmp_vec);
//        }
//        // Now loop over every possible shell atom index list and find 
//        for (int i=0; i<natom; i++) {
//          for (int j=0; j<natom; j++) {
//            for (int k=0; k<natom; k++) {
//              for (int l=0; l<natom; l++) {
//                std::vector<int> shell_atom_index_list{i,j,k,l};
//                int center_idx = 
//                int atom_idx = shell_atom_index_list[center_idx];
//
//                // Sort such that i <= j <= k ... so it corresponds to the upper triangle of totally symmetric nuclear derivative tensor
//                std::sort(multi_cart_idx.begin(), multi_cart_idx.end());
//                // Find flattened upper triangle nuclear derivative index
//                // Since the vector of vectors "cart_multidim_lookup" is sorted such that each vector is elementwise <= next vector, we can
//                // use binary search to find the flattened upper triangle 1d index that is in range of nderivs_triu 
//                int nuc_idx = 0;
//                auto it = lower_bound(cart_multidim_lookup.begin(), cart_multidim_lookup.end(), multi_cart_idx);
//                if (it != cart_multidim_lookup.end()) nuc_idx = it - cart_multidim_lookup.begin();
//                mapping[z][i][j][k][l] = nuc_idx;
//              }
//            } 
//          } 
//        } 
//    }   
//    return mapping;
//}   

// Writes ERI derivatives to disk 
void eri_deriv_disk(int deriv_order) { 
    // Derivative order
    //int deriv_order = accumulate(deriv_vec.begin(), deriv_vec.end(), 0);

    // Number of unique nuclear derivatives of ERI's
    unsigned int nderivs_triu = how_many_derivs(natom, deriv_order);

    // Check to make sure you are not flooding the disk.
    double check = (nbf * nbf * nbf * nbf * nderivs_triu * 8) / (1e9);
    assert(check < 2 && "Disk space required for ints exceeds 2 GB. Are you sure you know what you are doing?");

    // Number of unique shell derivatives output by libint (number of indices in buffer)
    int nshell_derivs = how_many_derivs(4, deriv_order);

    // Create mapping from 1d buffer index (flattened upper triangle shell derivative index) to multidimensional index
    const std::vector<std::vector<int>> buffer_multidim_lookup = generate_multi_index_lookup(12, deriv_order);
    // Create mapping from 1d cartesian coodinate index (flattened upper triangle cartesian derivative index) to multidimensional index
    const std::vector<std::vector<int>> cart_multidim_lookup = generate_multi_index_lookup(natom * 3, deriv_order);

    // Create array mapping from 1d buffer index (flattened upper triangle shell derivative index) ---> 
    // to the 1d flattened upper triangle cartesian nuclear derivative index
    //std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>> mapping = get_index_map(deriv_order, 4);

    // Libint engine for computing shell quartet derivatives
    libint2::Engine eri_engine(libint2::Operator::coulomb,obs.max_nprim(),obs.max_l(), deriv_order);
    const auto& buf_vec = eri_engine.results(); // will point to computed shell sets

    // Begin HDF5 jargon. Define file name and dataset name within file
    const H5std_string file_name("eri_deriv.h5");
    const H5std_string dataset_name("eri_deriv");
    // Create H5 File and prepare to fill with 0.0's
    H5File* file = new H5File(file_name,H5F_ACC_TRUNC);
    double fillvalue = 0.0;
    DSetCreatPropList plist;
    plist.setFillValue(PredType::NATIVE_DOUBLE, &fillvalue);
    // Create dataspace for file array (rank, dimensions)
    hsize_t file_dims[] = {nbf, nbf, nbf, nbf, nderivs_triu};
    DataSpace fspace(5, file_dims);
    // Create dataset and write 0.0's into the file 
    DataSet* dataset = new DataSet(file->createDataSet(dataset_name, PredType::NATIVE_DOUBLE, fspace, plist));
    // End HDF5 jargon.

    for(auto s1=0; s1!=obs.size(); ++s1) {
        auto bf1 = shell2bf[s1];     // Index of first basis function in shell 1
        auto atom1 = shell2atom[s1]; // Atom index of shell 1
        auto n1 = obs[s1].size();    // number of basis functions in shell 1
        for(auto s2=0; s2!=obs.size(); ++s2) {
            auto bf2 = shell2bf[s2];     // Index of first basis function in shell 2
            auto atom2 = shell2atom[s2]; // Atom index of shell 2
            auto n2 = obs[s2].size();    // number of basis functions in shell 2
            for(auto s3=0; s3!=obs.size(); ++s3) {
                auto bf3 = shell2bf[s3];     // Index of first basis function in shell 3
                auto atom3 = shell2atom[s3]; // Atom index of shell 3
                auto n3 = obs[s3].size();    // number of basis functions in shell 3
                for(auto s4=0; s4!=obs.size(); ++s4) {
                    auto bf4 = shell2bf[s4];     // Index of first basis function in shell 4
                    auto atom4 = shell2atom[s4]; // Atom index of shell 4
                    auto n4 = obs[s4].size();    // number of basis functions in shell 4

                    // If the atoms are the same we ignore it as the derivatives will be zero.
                    if (atom1 == atom2 && atom1 == atom3 && atom1 == atom4) continue;
                    std::vector<long> shell_atom_index_list{atom1,atom2,atom3,atom4};

                    eri_engine.compute(obs[s1], obs[s2], obs[s3], obs[s4]); // Compute shell set

                    // Define shell set slab, with extra dimension for unique derivatives, initialized with 0.0's
                    double shellset_slab [n1][n2][n3][n4][nderivs_triu] = {};

                    // For every unique shell set derivative in buffer represented by a flattened upper triangle index,
                    // map the index to its multi-dim shell derivative index, then to its multi-dim cartesian derivative indices, 
                    // then its flattened 1d cartesian derivative index which is the address in 'nderivs_triu' dimension of shellset_slab
                    for(auto i=0; i<nshell_derivs; ++i) {
                        auto ints_shellset = buf_vec[i]; // Location of the computed integrals
                        if (ints_shellset == nullptr)
                            continue;  // nullptr returned if the entire shell-set was screened out
                        //TODO eventually can replace below logic with:
                        //int nuc_idx = mapping[i][atom1][atom2][atom3][atom4];

                        // TODO finish writing function for this block of code
                        // Find multidimensional shell derivative index
                        auto multi_shell_idx = buffer_multidim_lookup[i];
                        // Find multidimensional cartesian coordinate derivative index 
                        std::vector<int> multi_cart_idx;
                        for (auto shell_idx : multi_shell_idx) {
                            //int center_idx, comp_idx = std::div(shell_idx, 3); // center_idx is 0,1,2,or 3. comp_idx is 0,1,2 <--> x,y,z
                            div_t tmp = std::div(shell_idx, 3); // center_idx is 0,1,2,or 3. comp_idx is 0,1,2 <--> x,y,z
                            int center_idx = tmp.quot;
                            int comp_idx = tmp.rem;
                            int atom_idx = shell_atom_index_list[center_idx];
                            multi_cart_idx.push_back(3 * atom_idx + comp_idx);
                        }
                        // Sort such that i <= j <= k ... so it corresponds to the upper triangle of totally symmetric nuclear derivative tensor
                        std::sort(multi_cart_idx.begin(), multi_cart_idx.end());

                        // Find flattened upper triangle nuclear derivative index
                        // Since the vector of vectors "cart_multidim_lookup" is sorted such that each vector is elementwise <= next vector, we can
                        // use binary search to find the flattened upper triangle 1d index that is in range of nderivs_triu 
                        int nuc_idx = 0;
                        auto it = lower_bound(cart_multidim_lookup.begin(), cart_multidim_lookup.end(), multi_cart_idx);
                        if (it != cart_multidim_lookup.end()) nuc_idx = it - cart_multidim_lookup.begin();
                        // TODO finish writing function for this block of code


                        // Loop over shell block, keeping a total count idx for the size of shell set
                        for(auto f1=0, idx=0; f1!=n1; ++f1) {
                            for(auto f2=0; f2!=n2; ++f2) {
                                for(auto f3=0; f3!=n3; ++f3) {
                                    for(auto f4=0; f4!=n4; ++f4, ++idx) {
                                        shellset_slab[f1][f2][f3][f4][nuc_idx] += ints_shellset[idx];
                                    }
                                }
                            }
                        }
                    }
                    // Now write this shell set slab to HDF5 file
                    hsize_t count[5] = {n1, n2, n3, n4, nderivs_triu};
                    hsize_t start[5] = {bf1, bf2, bf3, bf4, 0};
                    hsize_t stride[5] = {1,1,1,1,1}; // stride and block can be used to 
                    hsize_t block[5] = {1,1,1,1,1};  // add values to multiple places
                    fspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);
                    // Create dataspace defining for memory dataset to write to file
                    hsize_t mem_dims[] = {n1, n2, n3, n4, nderivs_triu};
                    DataSpace mspace(5, mem_dims);
                    start[0] = 0;
                    start[1] = 0;
                    start[2] = 0;
                    start[3] = 0;
                    start[4] = 0;
                    mspace.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);
                    // Write buffer data 'shellset_slab' with data type double from memory dataspace `mspace` to file dataspace `fspace`
                    dataset->write(shellset_slab, PredType::NATIVE_DOUBLE, mspace, fspace);
                }
            }
        }
    }
    // closes the dataset and file
    delete dataset;
    delete file;
}

PYBIND11_MODULE(pylibint, m) {
    m.def("initialize", &initialize, "Initializes libint, builds geom and basis, assigns globals");
    m.def("finalize", &finalize, "Kills libint");
    m.def("overlap", &overlap, "Computes overlap integrals with libint and return NumPy array"); 
    m.def("overlap_disk", &overlap_disk, "Computes overlap integrals and writes them to disk with HDF5");
    m.def("eri_disk", &eri_disk, "Computes coulomb integrals and writes them to disk with HDF5");
    m.def("eri_deriv_disk", &eri_deriv_disk, "Computes coulomb integral derivatives and writes them to disk with HDF5");
}


