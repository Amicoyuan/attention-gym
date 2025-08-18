## Tests

The purpose of the test module is to help you verify the precision error with cuda at the kernel level.

The test module has additional dependencies that need to be installed by the user

```
#pytest
pip install pytest

#sage_attention
git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention 
export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32 # parallel compiling (Optional)
python setup.py install  # or pip install -e .

#sparge_attention
git clone https://github.com/thu-ml/SpargeAttn.git
cd SpargeAttn
pip install ninja   # for parallel compilation
python setup.py install   # or pip install -e .
```

