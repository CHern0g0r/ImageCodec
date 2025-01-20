import os

import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import Extension, setup

cpp_args = ['-std=c++11']

functions_module = Extension(
    name="EntropyCodec",
    sources=["wrapper.cpp"],
    include_dirs=[pybind11.get_include()],
    language='c++',
    extra_compile_args=cpp_args,
    extra_link_args=['-lsupc++']
)

# functions_module = Pybind11Extension(
#     "EntropyCodec",
#     ["wrapper.cpp"],
# )


# functions_module = Extension(
#     name="EntropyCodec",
#     sources=["wrapper.cpp"],
#     include_dirs=[
#         os.path.join(pybind11.__path__[0], "include"),
#     ],
#     language='c++',
#     extra_link_args=['supc++']
# )

# setup(ext_modules=[functions_module], options={"build_ext": {"build_lib": ".."}})
setup(ext_modules=[functions_module], cmdclass={"build_ext": build_ext})
