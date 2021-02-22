#include <iostream>
#include <libint2.hpp>
//TEMP TODO
#include <pybind11/pybind11.h>

int main() {
  std::cout << "Hello World\n";
  return 0;
}

// PyBind needs -shared and -fPIC with Python and pybind11 includes.
// Could it be a conflict between libint version of python and pybind?? nahhhh

// I get segfaults with this routine:
// g++ -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2  -lint2 -L/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/ -c libint_interface.cc -o libint_interface.o 

// g++ -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2  -lint2 -L/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/ libint_interface.o -o libint_interface`python3-config --extension-suffix`


// Keep in mind your reference routine works fine:

// g++ -c libint_interface.cc -o libint_interface.o -I /home/adabbott/anaconda3/envs/psijax/include/eigen3 -I /home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -I /home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2  -lint2 -L /home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/

// g++  libint_interface.o -o libint_interface -I /home/adabbott/anaconda3/envs/psijax/include/eigen3 -I /home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -I /home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2  -lint2 -L /home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/

// Let's try being real verbose with it:

// g++ -shared -std=c++11 -fPIC -I/home/adabbott/anaconda3/envs/psijax/include/python3.6m -I/home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11/include -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2  -lint2 -L/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/ -c libint_interface.cc -o libint_interface.o 

// g++ -shared -std=c++11 -fPIC -I/home/adabbott/anaconda3/envs/psijax/include/python3.6m -I/home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11/include -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2  -lint2 -L/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/ libint_interface.o -o libint_interface`python3-config --extension-suffix` 


// g++ -std=c++11 -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2  -lint2 -L/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/ -c libint_interface.cc -o libint_interface.o
// g++ -std=c++11 -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2  -lint2 -L/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/ libint_interface.o -o exec

// Okay... EVEN THIS DOESNT WORK,, whtat!!!!
// g++ -std=c++11 -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2  -lint2 -L/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/ -c libint_interface.cc -o libint_interface.o
// g++ -std=c++11 -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2  -lint2 -L/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/ libint_interface.o -o exec

// Let's try rearranging... ig? This worked before, minus the -std
// g++ -c libint_interface.cc -o libint_interface.o -std=c++11 -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2  -lint2 -L/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/ 
// g++ libint_interface.o -o exec -std=c++11 -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2  -lint2 -L/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/ 


// Okay, so -c and -o has to happen first. Lets try including all the pybind shit now
// Next we will try after flags -shared,etc
// Okay, this fails. lets try not using weird names? 
// g++ -c libint_interface.cc -o libint_interface.o -shared -std=c++11 -fPIC -I/home/adabbott/anaconda3/envs/psijax/include/python3.6m -I/home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11/include -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2  -lint2 -L/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/  

// g++ libint_interface.o -o exec -shared -std=c++11 -fPIC -I/home/adabbott/anaconda3/envs/psijax/include/python3.6m -I/home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11/include -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2  -lint2 -L/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/  


// abovve still fails, really doesnt like shared and fPIC... maybe i can break it up
// g++ -c libint_interface.cc -o libint_interface.o -std=c++11 -I/home/adabbott/anaconda3/envs/psijax/include/python3.6m -I/home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11/include -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2  -lint2 -L/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/  

// g++ libint_interface.o -o exec -std=c++11 -I/home/adabbott/anaconda3/envs/psijax/include/python3.6m -I/home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11/include -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2  -lint2 -L/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/  


// Now try only passing shared on the creation of executable?
// g++ -c libint_interface.cc -o libint_interface.o -fPIC -std=c++11 -I/home/adabbott/anaconda3/envs/psijax/include/python3.6m -I/home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11/include -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2  -lint2 -L/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/  

// g++ libint_interface.o -o exec -std=c++11 -shared -fPIC -I/home/adabbott/anaconda3/envs/psijax/include/python3.6m -I/home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11/include -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2  -lint2 -L/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/  

// Okay CHECKPOINT: this works, try it wil import pybind11?
//g++ -c libint_interface.cc -o libint_interface.o -fPIC -std=c++11 -I/home/adabbott/anaconda3/envs/psijax/include/python3.6m -I/home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11/include -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2  -lint2 -L/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/
//g++ libint_interface.o -o exec -std=c++11 -fPIC -I/home/adabbott/anaconda3/envs/psijax/include/python3.6m -I/home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11/include -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2  -lint2 -L/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/

// Above appears to work, now lets go back
// Nevermind, can you add -shared in there wihtout messing things up?

//g++ -c libint_interface.cc -o libint_interface.o -shared -fPIC -std=c++11 -I/home/adabbott/anaconda3/envs/psijax/include/python3.6m -I/home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11/include -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2  -lint2 -L/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/
//g++ libint_interface.o -o exec  -shared -std=c++11 -fPIC -I/home/adabbott/anaconda3/envs/psijax/include/python3.6m -I/home/adabbott/anaconda3/envs/psijax/lib/python3.6/site-packages/pybind11/include -I/home/adabbott/anaconda3/envs/psijax/include/eigen3 -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include -I/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/include/libint2  -lint2 -L/home/adabbott/Git/libint/BUILD/libint-2.7.0-beta.6/

