#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdlib.h>
#include <libint2.hpp>

namespace py = pybind11;

// You can code all functions directly into here. Example:
int add(int i, int j) {
    return i + j;
}

// Try making a dummy function which calls some libint stuff to be sure its owrking
int test(int i, int j) {
    using namespace libint2;
    using namespace std;
    string xyzfilename = "/home/adabbott/h2.xyz"; 
    ifstream input_file(xyzfilename);
    vector<Atom> atoms = read_dotxyz(input_file);
    BasisSet obs("cc-pVDZ", atoms);
    return 0;
}

//libint2::initalize();
// write functionfor computing TEI's
//libint2::finalize();

// Define module named 'libint_interface' which can be imported with python
// The second arg, 'm' defines a variable py::module_ which can be used to create
// bindings. the def() methods generates binding code that exposes new functions to Python.
PYBIND11_MODULE(libint_interface, m) {
    m.doc() = "pybind11 libint interface to molecular integrals"; // optional module docstring
    m.def("add", &add, "A function which adds two numbers");
    m.def("test", &test, "Dummy function which does libint2 stuff and returns 0");
}

// THIS WORKS. -O3 optimization fixes issues of linking both pybind and libint. very lucky?
//g++ -c libint_interface.cc -o libint_interface.o -O3 -fPIC -shared -std=c++11 -I/home/adabbott/anaconda3/envs/psijax/include/python3.6m -I/home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11/include -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2  -lint2 -L/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/
//g++ libint_interface.o -o libint_interface`python3-config --extension-suffix` -std=c++11 -O3 -fPIC -shared -I/home/adabbott/anaconda3/envs/psijax/include/python3.6m -I/home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11/include -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2  -lint2 -L/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/



// After compilation, should be able to access this funciton:
// >>> import psijax
// >>> psijax.external_integrals.libint_interface.add(1,2)
// 3


// TODO Define functions, figure out args stuff, e.g. py::array_t<double>
// also define in another .cc file containing these routines
// TODO figure out how to handle BasisSet objects here. 
//void compute_tei()
//void compute_tei_deriv()
//void compute_overlap()
//void compute_overlap_deriv()
//void compute_kinetic()
//void compute_kinetic_deriv()
//void compute_potential()
//void compute_potential_deriv()

//PYBIND11_PLUGIN(libint_tei) {
//    py::module m("libint_interface", "pybind11 interface to libint molecule integrals and their derivatives")
//    m.def("compute_tei", &compute_tei, "Compute two-electron integral array, shape (nbf,nbf,nbf,nbf)")
//    m.def("compute_tei_deriv", &compute_tei, "Compute partial derivative of two-electron integral array, shape (nbf,nbf,nbf,nbf)")
//    m.def("compute_overlap", &compute_overlap, "Compute overlap integral array, shape (nbf,nbf)")
//    m.def("compute_overlap_deriv", &compute_overlap, "Compute (nbf,nbf,nbf,nbf) nuclear partial derivative of two-electron integral array")
//    m.def("compute_kinetic", &compute_kinetic, "Compute (nbf,nbf,nbf,nbf) two-electron integral array")
//    m.def("compute_kinetic_deriv", &compute_kinetic, "Compute (nbf,nbf) nuclear partial derivative of two-electron integral array")
//
//    return m.ptr();
//}
