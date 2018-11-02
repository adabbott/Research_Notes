#include <stdio.h>
#include <stdlib.h> 
#include <Python.h>

// other people do this hmmm
//int main (int argc, char **argv){
int main (){
    // create a faux vector to simulate whatt nitrogen will do
    #define N 3
    //double *vec = malloc(3 * N * sizeof(double));
    double *vec = malloc(3 * sizeof(double));
    vec[0] = 1.1;
    vec[1] = 1.1;
    vec[2] = 1.1;
    //vec[3] = 1.1;
    //vec[4] = 1.1;
    //vec[5] = 1.1;
    //vec[6] = 1.1;
    //vec[7] = 1.1;
    //vec[8] = 1.1;


    // declare py objects
    PyObject *pName, *pModule, *pFunc, *pArg, *pDict;
    //PyObject *pName, *pModule, *pFunc, *pArg, *pDict;
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
    //pArg = PyTuple_New(1);                                  // Convert input data to tuple because for some reason PyObject_CallObject requires it
    pArg = PyTuple_New(2);                                  // Convert input data to tuple because for some reason PyObject_CallObject requires it
    PyTuple_SetItem(pArg, 0, PyFloat_FromDouble(1.1));       // Put desired input of function into tuple. (the tuple, position in tuple, value to put into tuple)
    PyTuple_SetItem(pArg, 1, PyFloat_FromDouble(1.2));       // Put desired input of function into tuple. (the tuple, position in tuple, value to put into tuple)
    //pArg = PyTuple_New(3);                                  // Convert input data to tuple because for some reason PyObject_CallObject requires it
    //PyTuple_SetItem(pArg, 0, PyFloat_FromDouble(vec[0]));       // Put desired input of function into tuple. (the tuple, position in tuple, value to put into tuple)
    //PyTuple_SetItem(pArg, 1, PyFloat_FromDouble(vec[1]));       // Put desired input of function into tuple. (the tuple, position in tuple, value to put into tuple)
    //PyTuple_SetItem(pArg, 2, PyFloat_FromDouble(vec[2]));       // Put desired input of function into tuple. (the tuple, position in tuple, value to put into tuple)
    //PyTuple_SetItem(pArg, 0, PyFloat_FromDouble(1.1));       // Put desired input of function into tuple. (the tuple, position in tuple, value to put into tuple)
    //PyTuple_SetItem(pArg, 1, PyFloat_FromDouble(1.1));       // Put desired input of function into tuple. (the tuple, position in tuple, value to put into tuple)
    //PyTuple_SetItem(pArg, 2, PyFloat_FromDouble(1.1));       // Put desired input of function into tuple. (the tuple, position in tuple, value to put into tuple)
//!!!!!!!!!!!!
    //free(vec);
                                                            // Note that since we are creating a python object, it doesnt care if we write 27, 27. 27.0, whatever
    //PyObject* result = PyObject_CallObject(pFunc, pArg);    // Call the function on the tuple argument and get the result as a python object
    //PyObject* result = PyObject_CallFunction(pFunc, "O", pArg);    // Call the function on the tuple argument and get the result as a python object
    //PyObject* result = PyObject_CallFunction(pFunc, (const char*)"(d,d,d)", vec[0], vec[1], vec[2]);    // Call the function on the tuple argument and get the result as a python object
    PyObject* result = PyObject_CallFunction(pFunc, (const char*)"(O)", pArg);    // Call the function on the tuple argument and get the result as a python object
    PyErr_PrintEx(1);
    //Py_XDECREF(pArg);
    double f = PyFloat_AsDouble(result);                    // Convert the result back to a C type. Careful, C types are very picky!
    Py_XDECREF(result);
    printf("Output of python function: %f\n", f);           // Print result to be sure
    Py_Finalize();

    free(vec);
    return 0;
}
