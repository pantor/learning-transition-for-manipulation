#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <opencv2/opencv.hpp>

#include <orthographic_image/orthographic_image.hpp>

namespace py = pybind11;


// borrowed in spirit from https://github.com/yati-sagade/opencv-ndarray-conversion
// MIT License


#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#if PY_VERSION_HEX >= 0x03000000
    #define PyInt_Check PyLong_Check
    #define PyInt_AsLong PyLong_AsLong
#endif

struct Tmp {
    const char * name;

    Tmp(const char * name ) : name(name) {}
};

Tmp info("return value");

bool NDArrayConverter::init_numpy() {
    // this has to be in this file, since PyArray_API is defined as static
    import_array1(false);
    return true;
}


static PyObject* opencv_error = 0;

static int failmsg(const char *fmt, ...) {
    char str[1000];

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(str, sizeof(str), fmt, ap);
    va_end(ap);

    PyErr_SetString(PyExc_TypeError, str);
    return 0;
}

class PyAllowThreads {
public:
    PyAllowThreads() : _state(PyEval_SaveThread()) {}
    ~PyAllowThreads() {
        PyEval_RestoreThread(_state);
    }
private:
    PyThreadState* _state;
};

class PyEnsureGIL {
public:
    PyEnsureGIL() : _state(PyGILState_Ensure()) {}
    ~PyEnsureGIL() {
        PyGILState_Release(_state);
    }
private:
    PyGILState_STATE _state;
};

#define ERRWRAP2(expr) \
try { \
    PyAllowThreads allowThreads; \
    expr; \
} catch (const cv::Exception &e) { \
    PyErr_SetString(opencv_error, e.what()); \
    return 0; \
}

using namespace cv;

class NumpyAllocator: public MatAllocator {
public:
    NumpyAllocator() {
        stdAllocator = Mat::getStdAllocator();
    }

    ~NumpyAllocator() {}

    UMatData* allocate(PyObject* o, int dims, const int* sizes, int type, size_t* step) const {
        UMatData* u = new UMatData(this);
        u->data = u->origdata = (uchar*)PyArray_DATA((PyArrayObject*) o);
        npy_intp* _strides = PyArray_STRIDES((PyArrayObject*) o);
        for( int i = 0; i < dims - 1; i++ )
            step[i] = (size_t)_strides[i];
        step[dims-1] = CV_ELEM_SIZE(type);
        u->size = sizes[0]*step[0];
        u->userdata = o;
        return u;
    }

    UMatData* allocate(int dims0, const int* sizes, int type, void* data, size_t* step, int flags, UMatUsageFlags usageFlags) const {
        if( data != 0 ) {
            CV_Error(Error::StsAssert, "The data should normally be NULL!");
            // probably this is safe to do in such extreme case
            return stdAllocator->allocate(dims0, sizes, type, data, step, flags, usageFlags);
        }
        PyEnsureGIL gil;

        int depth = CV_MAT_DEPTH(type);
        int cn = CV_MAT_CN(type);
        const int f = (int)(sizeof(size_t)/8);
        int typenum = depth == CV_8U ? NPY_UBYTE : depth == CV_8S ? NPY_BYTE :
        depth == CV_16U ? NPY_USHORT : depth == CV_16S ? NPY_SHORT :
        depth == CV_32S ? NPY_INT : depth == CV_32F ? NPY_FLOAT :
        depth == CV_64F ? NPY_DOUBLE : f*NPY_ULONGLONG + (f^1)*NPY_UINT;
        int i, dims = dims0;
        cv::AutoBuffer<npy_intp> _sizes(dims + 1);
        for( i = 0; i < dims; i++ )
            _sizes[i] = sizes[i];
        if( cn > 1 )
            _sizes[dims++] = cn;
        PyObject* o = PyArray_SimpleNew(dims, _sizes, typenum);
        if(!o)
            CV_Error_(Error::StsError, ("The numpy array of typenum=%d, ndims=%d can not be created", typenum, dims));
        return allocate(o, dims0, sizes, type, step);
    }

    bool allocate(UMatData* u, int accessFlags, UMatUsageFlags usageFlags) const {
        return stdAllocator->allocate(u, accessFlags, usageFlags);
    }

    void deallocate(UMatData* u) const {
        if(!u)
            return;
        PyEnsureGIL gil;
        CV_Assert(u->urefcount >= 0);
        CV_Assert(u->refcount >= 0);
        if(u->refcount == 0) {
            PyObject* o = (PyObject*)u->userdata;
            Py_XDECREF(o);
            delete u;
        }
    }

    const MatAllocator* stdAllocator;
};

NumpyAllocator g_numpyAllocator;

bool NDArrayConverter::toMat(PyObject *o, Mat &m) {
    bool allowND = true;
    if(!o || o == Py_None) {
        if( !m.data )
            m.allocator = &g_numpyAllocator;
        return true;
    }

    if( PyInt_Check(o) ) {
        double v[] = {static_cast<double>(PyInt_AsLong((PyObject*)o)), 0., 0., 0.};
        m = Mat(4, 1, CV_64F, v).clone();
        return true;
    }
    if( PyFloat_Check(o) ) {
        double v[] = {PyFloat_AsDouble((PyObject*)o), 0., 0., 0.};
        m = Mat(4, 1, CV_64F, v).clone();
        return true;
    }
    if( PyTuple_Check(o) ) {
        int i, sz = (int)PyTuple_Size((PyObject*)o);
        m = Mat(sz, 1, CV_64F);
        for( i = 0; i < sz; i++ ) {
            PyObject* oi = PyTuple_GET_ITEM(o, i);
            if( PyInt_Check(oi) )
                m.at<double>(i) = (double)PyInt_AsLong(oi);
            else if( PyFloat_Check(oi) )
                m.at<double>(i) = (double)PyFloat_AsDouble(oi);
            else {
                failmsg("%s is not a numerical tuple", info.name);
                m.release();
                return false;
            }
        }
        return true;
    }

    if( !PyArray_Check(o) ) {
        failmsg("%s is not a numpy array, neither a scalar", info.name);
        return false;
    }

    PyArrayObject* oarr = (PyArrayObject*) o;

    bool needcopy = false, needcast = false;
    int typenum = PyArray_TYPE(oarr), new_typenum = typenum;
    int type = typenum == NPY_UBYTE ? CV_8U :
               typenum == NPY_BYTE ? CV_8S :
               typenum == NPY_USHORT ? CV_16U :
               typenum == NPY_SHORT ? CV_16S :
               typenum == NPY_INT ? CV_32S :
               typenum == NPY_INT32 ? CV_32S :
               typenum == NPY_FLOAT ? CV_32F :
               typenum == NPY_DOUBLE ? CV_64F : -1;

    if( type < 0 ) {
        if( typenum == NPY_INT64 || typenum == NPY_UINT64 || typenum == NPY_LONG ) {
            needcopy = needcast = true;
            new_typenum = NPY_INT;
            type = CV_32S;
        } else {
            failmsg("%s data type = %d is not supported", info.name, typenum);
            return false;
        }
    }

#ifndef CV_MAX_DIM
    const int CV_MAX_DIM = 32;
#endif

    int ndims = PyArray_NDIM(oarr);
    if(ndims >= CV_MAX_DIM) {
        failmsg("%s dimensionality (=%d) is too high", info.name, ndims);
        return false;
    }

    int size[CV_MAX_DIM+1];
    size_t step[CV_MAX_DIM+1];
    size_t elemsize = CV_ELEM_SIZE1(type);
    const npy_intp* _sizes = PyArray_DIMS(oarr);
    const npy_intp* _strides = PyArray_STRIDES(oarr);
    bool ismultichannel = ndims == 3 && _sizes[2] <= CV_CN_MAX;

    for( int i = ndims-1; i >= 0 && !needcopy; i-- ) {
        // these checks handle cases of
        //  a) multi-dimensional (ndims > 2) arrays, as well as simpler 1- and 2-dimensional cases
        //  b) transposed arrays, where _strides[] elements go in non-descending order
        //  c) flipped arrays, where some of _strides[] elements are negative
        // the _sizes[i] > 1 is needed to avoid spurious copies when NPY_RELAXED_STRIDES is set
        if( (i == ndims-1 && _sizes[i] > 1 && (size_t)_strides[i] != elemsize) ||
            (i < ndims-1 && _sizes[i] > 1 && _strides[i] < _strides[i+1]) )
            needcopy = true;
    }

    if (ismultichannel && _strides[1] != (npy_intp)elemsize*_sizes[2]) {
        needcopy = true;
    }

    if (needcopy) {
        //if (info.outputarg)
        //{
        //    failmsg("Layout of the output array %s is incompatible with cv::Mat (step[ndims-1] != elemsize or step[1] != elemsize*nchannels)", info.name);
        //    return false;
        //}

        if (needcast) {
            o = PyArray_Cast(oarr, new_typenum);
            oarr = (PyArrayObject*) o;
        }
        else {
            oarr = PyArray_GETCONTIGUOUS(oarr);
            o = (PyObject*) oarr;
        }

        _strides = PyArray_STRIDES(oarr);
    }

    // Normalize strides in case NPY_RELAXED_STRIDES is set
    size_t default_step = elemsize;
    for (int i = ndims - 1; i >= 0; --i) {
        size[i] = (int)_sizes[i];
        if (size[i] > 1) {
            step[i] = (size_t)_strides[i];
            default_step = step[i] * size[i];
        } else {
            step[i] = default_step;
            default_step *= size[i];
        }
    }

    // handle degenerate case
    if (ndims == 0) {
        size[ndims] = 1;
        step[ndims] = elemsize;
        ndims++;
    }

    if (ismultichannel) {
        ndims--;
        type |= CV_MAKETYPE(0, size[2]);
    }

    if (ndims > 2 && !allowND) {
        failmsg("%s has more than 2 dimensions", info.name);
        return false;
    }

    m = Mat(ndims, size, type, PyArray_DATA(oarr), step);
    m.u = g_numpyAllocator.allocate(o, ndims, size, type, step);
    m.addref();

    if (!needcopy) {
        Py_INCREF(o);
    }
    m.allocator = &g_numpyAllocator;

    return true;
}

PyObject* NDArrayConverter::toNDArray(const cv::Mat& m) {
    if (!m.data)
        Py_RETURN_NONE;
    Mat temp, *p = (Mat*)&m;
    if (!p->u || p->allocator != &g_numpyAllocator) {
        temp.allocator = &g_numpyAllocator;
        ERRWRAP2(m.copyTo(temp));
        p = &temp;
    }
    PyObject* o = (PyObject*)p->u->userdata;
    Py_INCREF(o);
    return o;
}


PYBIND11_MODULE(orthographical, m) {
  NDArrayConverter::init_numpy();

  py::class_<OrthographicImage>(m, "OrthographicImage")
    .def(py::init<const cv::Mat&, double, double, double>())
    .def(py::init<const cv::Mat&, double, double, double, const std::string&>())
    .def(py::init<const cv::Mat&, double, double, double, const std::string&, Affine>())
    .def_readwrite("mat", &OrthographicImage::mat)
    .def_readwrite("pixel_size", &OrthographicImage::pixel_size)
    .def_readwrite("min_depth", &OrthographicImage::min_depth)
    .def_readwrite("max_depth", &OrthographicImage::max_depth)
    .def_readwrite("camera", &OrthographicImage::camera)
    .def_readwrite("pose", &OrthographicImage::pose)
    .def("depth_from_value", &OrthographicImage::depthFromValue)
    .def("value_from_depth", &OrthographicImage::valueFromDepth)
    .def("project", &OrthographicImage::project)
    .def("position_from_index", &OrthographicImage::positionFromIndex)
    .def("index_from_position", &OrthographicImage::indexFromPosition)
    .def("translate", &OrthographicImage::translate)
    .def("rotate_x", &OrthographicImage::rotateX)
    .def("rotate_y", &OrthographicImage::rotateY)
    .def("rotate_z", &OrthographicImage::rotateZ)
    .def("rescale", &OrthographicImage::rescale);
    /*.def(py::pickle(
      [](const OrthographicImage &p) { // __getstate__
        return py::make_tuple(p.mat.clone(), p.pixel_size, p.min_depth, p.max_depth, p.camera);
      },
      [](py::tuple t) { // __setstate__
        if (t.size() != 4)
          throw std::runtime_error("Invalid state!");

        OrthographicImage p(t[0].cast<cv::Mat>(), t[1].cast<double>(), t[2].cast<double>(), t[3].cast<double>(), t[4].cast<std::string>());
        return p;
      }
    )); */
}
