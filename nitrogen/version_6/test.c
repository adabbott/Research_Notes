#include <stdio.h>
#include <stdlib.h> 
#include <Python.h>

#define N 3

int getN(){
    return N;
}

double calcSurface(double *X){
    // declare Python objects
    PyObject *pName, *pModule, *pFunc, *pArg, *pDict, *pReturn;
    Py_Initialize();
    pName = PyUnicode_FromString("my_module");
    if (!pName)
    {
        printf("pName\n");
        return 0;
    }
    pModule = PyImport_Import(pName);                       // import module
    pDict = PyModule_GetDict(pModule);                      // get dictionary mapping between the NAME of every python object and the objects themselves 
    pFunc = PyDict_GetItemString(pDict, (char*)"my_func");  // Get the Python object corresponding to key in pDict, "my_func", which in this case is a function object
    pArg = PyTuple_New(3 * N);                                  // Convert input data to tuple because for some reason PyObject_CallObject requires it
    // Put desired input of function into tuple. (the tuple, position in tuple, value to put into tuple)
    int i;
    for(i=0; i < 3 * N; i++)
    {
    PyTuple_SetItem(pArg, i, PyFloat_FromDouble(X[i]));
    }
    // Call the function on the tuple argument and get the result as a python object
    // Arguments are (python function, () means tuple, O means python object, pArg is python object
    pReturn = PyObject_CallFunction(pFunc, (const char*)"(O)", pArg);
    double value = PyFloat_AsDouble(pReturn); 
    // Dereference all python objects. Their memory will be cleared. Probably necessary. Doesn't hurt.
    Py_XDECREF(pName);  Py_XDECREF(pModule); Py_XDECREF(pFunc); Py_XDECREF(pArg); Py_XDECREF(pDict); Py_XDECREF(pReturn);
    printf("Output of python function: %f\n", value);           // Print result
    Py_Finalize();                                          // exit python interpreter

    // Return value
    return value;
}

int main (){
    // create a faux vector to simulate what nitrogen will do 
    double *vec = malloc(3 * N * sizeof(double));
    vec[0] = 2.1;
    vec[1] = 9.1;
    vec[2] = 1.8;
    vec[3] = 2.1;
    vec[4] = 2.0;
    vec[5] = 1.1;
    vec[6] = 4.1;
    vec[7] = 1.1;
    vec[8] = 1.1;

    double r;
    r = calcSurface(vec);

    return 0;
}
