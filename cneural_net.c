// Also consider template at https://github.com/python/cpython/blob/main/Modules/xxmodule.c

#define PY_SSIZE_T_CLEAN
#include <Python.h>


static PyObject *
cneural_net_system(PyObject *self, PyObject *args)
{
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    // sts = system(command);
    return PyLong_FromLong(1);
}


static PyMethodDef cneural_net_methods[] = {
    {"system",  cneural_net_system, METH_VARARGS, "Blah"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


static struct PyModuleDef cneural_net_module = {
    PyModuleDef_HEAD_INIT,
    "cneural_net",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,
    cneural_net_methods
};


PyMODINIT_FUNC
PyInit_cneural_net(void)
{
    return PyModule_Create(&cneural_net_module);
}
