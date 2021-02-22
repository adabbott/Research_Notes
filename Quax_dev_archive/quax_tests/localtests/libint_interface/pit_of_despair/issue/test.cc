#include <iostream>
#include <libint2.hpp>

int main() {
  std::cout << "Libint2 successfully imported\n";
  return 0;
}



// Test static library installation claims:
// g++ /home/adabbott/Git/new_libint/libint-2.7.0-beta.3/

// g++ test.cc -o test -I /home/adabbott/anaconda3/envs/psijax/include/eigen3 -I /home/adabbott/Git/new_libint/libint-2.7.0-beta.6/PREFIX/include -I /home/adabbott/Git/new_libint/libint-2.7.0-beta.6/PREFIX/include/libint2 -lint2 -L /home/adabbott/Git/new_libint/libint-2.7.0-beta.6/PREFIX/lib

// vs two step:

// g++ -c test.cc -o test.o -I /home/adabbott/anaconda3/envs/psijax/include/eigen3 -I /home/adabbott/Git/new_libint/libint-2.7.0-beta.6/PREFIX/include -I /home/adabbott/Git/new_libint/libint-2.7.0-beta.6/PREFIX/include/libint2 -lint2 -L /home/adabbott/Git/new_libint/libint-2.7.0-beta.6/PREFIX/lib

// g++ test.o -o test -I /home/adabbott/anaconda3/envs/psijax/include/eigen3 -I /home/adabbott/Git/new_libint/libint-2.7.0-beta.6/PREFIX/include -I /home/adabbott/Git/new_libint/libint-2.7.0-beta.6/PREFIX/include/libint2 -lint2 -L /home/adabbott/Git/new_libint/libint-2.7.0-beta.6/PREFIX/lib



// g++ -I /home/adabbott/anaconda3/envs/psijax/include/eigen3 -I /home/adabbott/Git/new_libint/libint-2.7.0-beta.6/PREFIX/include -I /home/adabbott/Git/new_libint/libint-2.7.0-beta.6/PREFIX/include/libint2 -lint2 -L /home/adabbott/Git/new_libint/libint-2.7.0-beta.6/PREFIX/lib -c test.cc -o test.o


// g++ -I /home/adabbott/anaconda3/envs/psijax/include/eigen3 -I /home/adabbott/Git/new_libint/libint-2.7.0-beta.6/PREFIX/include -I /home/adabbott/Git/new_libint/libint-2.7.0-beta.6/PREFIX/include/libint2 -lint2 -L /home/adabbott/Git/new_libint/libint-2.7.0-beta.6/PREFIX/lib  test.o -o test
