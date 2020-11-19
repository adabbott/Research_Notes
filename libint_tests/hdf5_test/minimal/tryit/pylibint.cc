#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdlib.h>
#include <libint2.hpp>
#include <H5Cpp.h>

namespace py = pybind11;

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

PYBIND11_MODULE(pylibint, m) {
    m.def("initialize", &initialize, "Initializes libint, builds geom and basis, assigns globals");
    m.def("finalize", &finalize, "Kills libint");
    m.def("overlap", &overlap, "Computes overlap integrals with libint");
}

