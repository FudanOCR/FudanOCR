from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

ext_module = Extension(
    "nms",
    sources=["nms.pyx"],
    extra_compile_args=["-std=c++11"],
    language="c++",
    include_dirs=[numpy.get_include()]
)

setup(ext_modules=cythonize(ext_module,
                            language_level=3,
                            annotate=True))
