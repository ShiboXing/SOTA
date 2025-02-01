from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, include_paths

lstm_ext = CUDAExtension(
    name="lstm_cell",
    sources=[
        "lstm.cpp",
        "lstm_cell_fwd.cu"
    ],
    include_dirs=include_paths() + [
        "/root/pydev/include/python3.12/",
        "/usr/local/cuda-12.1/targets/x86_64-linux/include/",
    ],
)


setup(
    name="lstm_cell",
    ext_modules=[lstm_ext],
    cmdclass={"build_ext": BuildExtension},
)
