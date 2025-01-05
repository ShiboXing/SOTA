from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name="lstm_cell",
    ext_modules=[cpp_extension.CppExtension("lstm_cell", ["lstm.cpp"])],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)

Extension(
    name="lstm_cell",
    sources=["lstm.cpp"],
    include_dirs=cpp_extension.include_paths() + \
        ["/usr/include/python3.12/"],
    language="c++",
)
