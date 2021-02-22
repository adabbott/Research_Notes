#include <iostream>

//void libint2_static_init();
//void libint2_static_cleanup();
#include <libint2.hpp>
//#include <libint2_iface.h>
//#include <libint2_params.h>
//#include <libint2.h>
 

int main() {
  std::cout << "Hello World\n";
  return 0;
}

// Compile statement:
// g++ -std=c++11 -I/usr/local/include/ -L/usr/local/lib/ -I/usr/include/eigen3 test.cpp -o test.o

// g++ -std=c++11 -I/usr/local/include/libint2 -I/usr/local/include -L/usr/local/lib /usr/local/lib/libint2.a -lint2 -I/usr/include/eigen3 test.cpp -o test.o

// CMake hartree fock files use this:
// g++ -std=c++11 -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2 test.cpp
