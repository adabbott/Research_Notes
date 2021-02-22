#include <iostream>
#include <stdlib.h>
#include <vector>

using namespace std;

int main() {
  std::cout << "Hello World\n";
  int atom1 = 2;
  int atom2 = 0;
  int atom3 = 1;
  int atom4 = 2;

  std::vector<int> deriv_vec = {1,0,0,1,0,0,1,0,0};

  std::vector<int> desired_atom_indices;
  std::vector<int> desired_coordinates;
  for (int i = 0; i < deriv_vec.size(); i++) {
      if (deriv_vec[i] > 0) {
          for (int j = 0; j < deriv_vec[i]; j++) {
              desired_atom_indices.push_back(i / 3);
              desired_coordinates.push_back(i % 3);
          }
      }
  }

  for (auto i = desired_atom_indices.begin(); i != desired_atom_indices.end(); ++i)
     std::cout << *i << ' ';
  for (auto i = desired_coordinates.begin(); i != desired_coordinates.end(); ++i)
     std::cout << *i << ' ';
  //std::cout << desired_atom_indices;
  //std::cout << desired_coordinates;


  //int natom = 4;
  //bool tmp = true;
  //std::vector<bool> desired_atoms = {false, false, true, true};
  //for (int i = 0; i < natom; ++i) {
  //   if (desired_atoms[i]) {
  //       tmp = (tmp && (i == atom1 || i == atom2 || i == atom3 || i == atom4));
  //   }
  //}

  //std::cout << tmp;
  //std::cout << "Hello World\n";
//  std::cout tmp;

}

  //vector<int> vect{ 10, 20, 30 }; 
//
//
//
//  //std::vector<bool> desired_atoms(natom, false);
//  //for (int i = 0; i < natom; i++) {
//  //    bool desired = false;
//  //    // x
//  //    if (orders[3 * i + 0] > 0) desired = true;
//  //    // y
//  //    if (orders[3 * i + 1] > 0) desired = true;
//  //    // z
//  //    if (orders[3 * i + 2] > 0) desired = true;
//  //    desired_atoms[i] = desired;
//  //}
//
//  //std::vector<bool> desired_atoms = {true, true, true, false};
//  //std::vector<bool> m_allFalse = {false, false, false, false, false};
//  //  vector<bool> one {
//  //for (int i = 0; i < natom; ++i) {
//  //   if (desired_atoms[i]);
//  //       tmp = (tmp && (i == atom1 || i == atom2 || i == atom3 || i == atom4));
//  //}
//  return 0;
//}

//using namespace std; 
//  
//int main() 
//{ 
//static const int arr[] = {16,2,77,29};
////vector<int> vec (arr, arr + sizeof(arr) / sizeof(arr[0]) );
//    //vector<int> vect = { 10, 20, 30 }; 
//    return 0; 
//} 
//} 
