#include "python_extension.hpp"

// @param args: contains a single string object which is a JSON string containing the algorithm configuration
static PyObject * configure(PyObject * self, PyObject * args) {
    const char * configuration;
    if (!PyArg_ParseTuple(args, "s", & configuration)) { return NULL; }

    std::istringstream config_stream(configuration);
    GOSDT::configure(config_stream);

    return Py_BuildValue("");
}

// @param args: contains a single string object which contains the training data in CSV form
// @returns a string object containing a JSON array of all resulting models
static PyObject * fit(PyObject * self, PyObject * args) {
    const char * dataset;
    if (!PyArg_ParseTuple(args, "s", & dataset)) { return NULL; }

    std::istringstream data_stream(dataset);
    GOSDT model;
    std::string result;
    model.fit(data_stream, result);

    return Py_BuildValue("s", result.c_str());
}

// @returns the number of seconds spent training
static PyObject * time(PyObject * self, PyObject * args) { return Py_BuildValue("f", GOSDT::time); }

// @returns the number of iterations spent training
static PyObject * iterations(PyObject * self, PyObject * args) { return Py_BuildValue("i", GOSDT::iterations); }

// @returns the number of vertices in the depency graph
static PyObject * size(PyObject * self, PyObject * args) { return Py_BuildValue("i", GOSDT::size); }

// @returns the current status code
static PyObject * status(PyObject * self, PyObject * args) { return Py_BuildValue("i", GOSDT::status); }

// @returns the global lower bound at the end of training
static PyObject * lower_bound(PyObject * self, PyObject * args) { return Py_BuildValue("d", GOSDT::lower_bound); }

// @returns the global upper bound at the end of training
static PyObject * upper_bound(PyObject * self, PyObject * args) { return Py_BuildValue("d", GOSDT::upper_bound); }

// @returns the loss of the tree found at the end of training (or trees - if more than one was found they should have the same loss)
static PyObject * model_loss(PyObject * self, PyObject * args) { return Py_BuildValue("f", GOSDT::model_loss); }

// Define the list of methods for a module
static PyMethodDef libosrt_methods[] = {
    // { method name, method pointer, method parameter format, method description }
    {"configure", configure, METH_VARARGS, "Configures the algorithm using an input JSON string"},
    {"fit", fit, METH_VARARGS, "Trains the model using an input CSV string"},
    {"time", time, METH_NOARGS, "Number of seconds spent training"},
    {"iterations", iterations, METH_NOARGS, "Number of iterations spent training"},
    {"size", size, METH_NOARGS, "Number of vertices in the depency graph"},
    {"status", status, METH_NOARGS, "Check the status code of the algorithm"},
    {"lower_bound", lower_bound, METH_NOARGS, "Check the lower_bound code of the algorithm"},
    {"upper_bound", upper_bound, METH_NOARGS, "Check the upper_bound code of the algorithm"},
    {"model_loss", model_loss, METH_NOARGS, "Check the model_loss code of the algorithm"},
    {NULL, NULL, 0, NULL}
};

// Define the module
static struct PyModuleDef libosrt = {
    PyModuleDef_HEAD_INIT,
    "libosrt", // Module Name
    "Optimal Sparse Regression Trees", // Module Description
    -1, // Size of per-interpreter state
    libosrt_methods // Module methods
};

// Initialize the module
PyMODINIT_FUNC PyInit_libosrt(void) {
    return PyModule_Create(&libosrt);
}