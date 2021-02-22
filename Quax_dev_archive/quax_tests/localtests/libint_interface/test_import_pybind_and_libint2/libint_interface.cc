#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdlib.h>
#include <libint2.hpp>

namespace py = pybind11;

//libint2::initalize();
// write functionfor computing TEI's
//libint2::finalize();

// You can code all functions directly into here. Example:
int add(int i, int j) {
    return i + j;
}

// Define module named 'libint_interface' which can be imported with python
// The second arg, 'm' defines a variable py::module_ which can be used to create
// bindings. the def() methods generates binding code that exposes new functions to Python.
PYBIND11_MODULE(libint_interface, m) {
    m.doc() = "pybind11 libint interface to molecular integrals"; // optional module docstring
    m.def("add", &add, "A function which adds two numbers");
}

// Compile with the following:
// c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` libint_interface.cpp -o libint_interface`python3-config --extension-suffix`
// Will need to add to compile line, once libint is added:
// -I/libint_prefix/include -L/libint_prefix/lib
// FOR NOW, until you get a better libint compiled:
// -I/home/adabbott/Git/dummy_libint/libint-2.7.0-beta.3/include  -L/home/adabbott/Git/dummy_libint/libint-2.7.0-beta.3/lib
// -I/home/adabbott/anaconda3/envs/psijax/include/eigen3

// New compilation: Two steps
// g++ -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` -c libint_interface.cc -o libint_interface.o 
// g++ -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` libint_interface.o -o libint_interface`python3-config --extension-suffix`

// Okay, to add libint support:
// g++ -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2  -lint2 -L/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/ -c libint_interface.cc -o libint_interface.o 

// g++ -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2  -lint2 -L/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/ libint_interface.o -o libint_interface`python3-config --extension-suffix`


// g++ -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` -c libint_interface.cc -o libint_interface`python3-config --extension-suffix`


// This works on a hello world, with first importing libint then pybind/pybind.h
// Try it:
//g++ -c libint_interface.cc -o libint_interface.o -fPIC -std=c++11 -I/home/adabbott/anaconda3/envs/psijax/include/python3.6m -I/home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11/include -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2  -lint2 -L/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/
//g++ libint_interface.o -o exec -std=c++11 -fPIC -I/home/adabbott/anaconda3/envs/psijax/include/python3.6m -I/home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11/include -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2  -lint2 -L/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/

// Above fails at linking.  

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
