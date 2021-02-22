#include <iostream>
//#include <libint2.hpp>

int main() {
  std::cout << "Hello";
  return 0;
}

// Compile statement:
// g++ -std=c++11 -I/usr/local/include/ -L/usr/local/lib/ -I/usr/include/eigen3 test.cpp -o test.o

// g++ -std=c++11 -L/usr/local/lib/ -I/usr/include/eigen3 test.cpp -o test.o
// g++ -std=c++11 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -I/usr/include/eigen3 -L/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/ test.cpp -o test.o
