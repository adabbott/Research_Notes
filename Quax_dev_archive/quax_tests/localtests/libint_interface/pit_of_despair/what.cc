#include <libint2.hpp>
#include <stdlib.h>
//#include <fstream>
//#include <string>
//#include <vector>
#include <iostream>

//using namespace libint2;
using namespace std;


//int main(int argc, char* argv[]) {
int main() {
  cout << "\nHello World!\n";
  //libint2::initialize();  
  //string xyzfilename = "/home/adabbott/Git/PsiTorch/PsiJax/localtests/libint_interface/tmp.xyz";
  //ifstream input_file(xyzfilename);
  //vector<Atom> atoms = read_dotxyz(input_file);
  //BasisSet obs("cc-pVDZ", atoms);
  //libint2::finalize();  
  return 0;
}



// c++ -O3 -Wall -shared -std=c++11 -fPIC -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 what.cc -o what.o
// Will need to add to compile line, once libint is added:
// -I/libint_prefix/include -L/libint_prefix/lib
// FOR NOW, until you get a better libint compiled:
// -I/home/adabbott/Git/dummy_libint/libint-2.7.0-beta.3/include  -L/home/adabbott/Git/dummy_libint/libint-2.7.0-beta.3/lib
// -I/home/adabbott/anaconda3/envs/psijax/include/eigen3

// c++ -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/usr/local/include -L/usr/local/lib  what.cc -o what.o


// c++ -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/usr/local/include -L/usr/local/lib -I/home/adabbott/Git/dummy_libint/libint-2.7.0-beta.3/include -L/home/adabbott/Git/dummy_libint/libint-2.7.0-beta.3/lib what.cc -o what.o

// g++ -std=c++11 -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/usr/local/include -L/usr/local/lib -I/usr/local/include/libint2 -I/home/adabbott/Git/dummy_libint/libint-2.7.0-beta.3 -I/home/adabbott/Git/dummy_libint/libint-2.7.0-beta.3/include -L/home/adabbott/Git/dummy_libint/libint-2.7.0-beta.3/lib comon.cc -o comon.o

// g++ -std=c++11 -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint_pgrad/libint-2.7.0-beta.6/PREFIX/include -I/home/adabbott/Git/libint_pgrad/libint-2.7.0-beta.6/PREFIX/include/libint2 -L/home/adabbott/Git/libint_pgrad/libint-2.7.0-beta.6/PREFIX/lib comon.cc

 

