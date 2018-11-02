#include <stdio.h>
#include <stdlib.h>
#include <Python.h>

// need to have these arguments hmmm what do these mean
int main (int argc, char **argv){
    // declare py objects
    PyObject *pName, *pModule, *pFunc, *pArg, *pDict, *pReturn;

    Py_Initialize();
    // ensure current directory is in path
    PyObject *sys = PyImport_ImportModule("sys");
    PyObject *path = PyObject_GetAttrString(sys, "path");
    PyList_Append(path, PyUnicode_FromString("."));

    // python script name is my_module.py
    pName = PyUnicode_FromString("my_module");
    if (!pName)
    {
        printf("pName\n");
        return 0;
    }

    pModule = PyImport_Import(pName);
    pDict = PyModule_GetDict(pModule);
    // function in my_module.py is called my_func. Define the Python function.
    pFunc = PyDict_GetItemString(pDict, (char*)"my_func");
    // PyObject_CallObject requires Tuple input
    pArg = PyTuple_New(1);
    PyTuple_SetItem(pArg, 0, PyFloat_FromDouble(50.0));
    PyObject* result = PyObject_CallObject(pFunc, pArg);
    // convert back to C type
    long f = PyLong_AsLong(result);
    printf("\nplease work: %d\n", f);
    Py_Finalize();
    return 0;
}
