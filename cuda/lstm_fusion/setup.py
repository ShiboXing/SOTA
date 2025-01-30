from setuptools import setup, Extension
from torch.utils import cpp_extension

lstm_ext = Extension(
    name="lstm_cell",
    sources=[
        "lstm.cpp",
        "lstm_cell_fwd.cu"
    ],
    include_dirs=cpp_extension.include_paths() + [
        "/root/pydev/include/python3.12/",
        "/usr/local/cuda-12.1/targets/x86_64-linux/include/",
    ],
    library_dirs=[
        "/usr/local/cuda/lib64",  # CUDA library path
    ],
    libraries=["cudart"],  # CUDA runtime library
    language="c++",
)


setup(
    name="lstm_cell",
    ext_modules=[lstm_ext],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
