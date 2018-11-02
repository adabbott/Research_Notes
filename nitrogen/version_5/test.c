#include <stdio.h>
#include <stdlib.h> 
#include <Python.h>

// other people do this hmmm
//int main (int argc, char **argv){
int main (){
    // create a faux vector to simulate what nitrogen will do
    #define N 3
    double *vec = malloc(3 * N * sizeof(double));
    vec[0] = 1.1;
    vec[1] = 1.1;
    vec[2] = 1.1;
    vec[3] = 1.1;
    vec[4] = 1.1;
    vec[5] = 1.1;
    vec[6] = 1.1;
    vec[7] = 1.1;
    vec[8] = 1.1;

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
    PyTuple_SetItem(pArg, i, PyFloat_FromDouble(vec[i]));
    }
    // Call the function on the tuple argument and get the result as a python object
    // Arguments are (python function, () means tuple, O means python object, pArg is python object
    pReturn = PyObject_CallFunction(pFunc, (const char*)"(O)", pArg);
    double f = PyFloat_AsDouble(pReturn); 
    // Dereference all python objects. Their memory will be cleared. Probably necessary. Doesn't hurt.
    Py_XDECREF(pName);  Py_XDECREF(pModule); Py_XDECREF(pFunc); Py_XDECREF(pArg); Py_XDECREF(pDict); Py_XDECREF(pReturn);
    printf("Output of python function: %f\n", f);           // Print result
    Py_Finalize();                                          // exit python interpreter
    free(vec);
    return 0;
}
