#include <stdio.h>
#include <stdlib.h>
#include <Python.h>

// other people do this hmmm
//int main (int argc, char **argv){
int main (){
    // declare py objects
    PyObject *pName, *pModule, *pFunc, *pArg, *pDict, *pReturn;
    Py_Initialize();
    // ensure current directory is in path
    // this apparently isnt necessary? maybe py3.7 solves issues with path?
    //PyObject *sys = PyImport_ImportModule("sys");
    //PyObject *path = PyObject_GetAttrString(sys, "path");
    //PyList_Append(path, PyUnicode_FromString("."));

    // python script name is my_module.py
    pName = PyUnicode_FromString("my_module");
    // I guess this checks to makes sure the module was found..  
    if (!pName)
    {
        printf("pName\n");
        return 0;
    }
    pModule = PyImport_Import(pName);                       // import module
    pDict = PyModule_GetDict(pModule);                      // get dictionary mapping between the NAME of every python object and the objects themselves 
    pFunc = PyDict_GetItemString(pDict, (char*)"my_func");  // Get the Python object corresponding to key in pDict, "my_func", which in this case is a function object
    pArg = PyTuple_New(1);                                  // Convert input data to tuple because for some reason PyObject_CallObject requires it
    PyTuple_SetItem(pArg, 0, PyFloat_FromDouble(10.0));     // Put desired input of function into tuple. (the tuple, position in tuple, value to put into tuple)
    PyObject* result = PyObject_CallObject(pFunc, pArg);    // Call the function on the tuple argument and get the result as a python object
    double f = PyFloat_AsDouble(result);                    // Convert the result back to a C type. Careful, C types are very picky!
    printf("\nplease work: %f\n", f);
    //long d = PyLong_AsLong(result);                       // If an integer, would need to do this:
    //printf("\nplease work: %d\n", d);
    Py_Finalize();
    return 0;
}
