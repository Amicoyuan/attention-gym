## Benchmarks

The purpose of the benchmarks module is to help you test the performance of the current triton kernel and compare it with the cuda kernel.


The benchmark module has additional dependencies that need to be installed by the user

```
#matplotlib
pip install matplotlib

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

