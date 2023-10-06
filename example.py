from numba import types
from numba_soa import compile_ptx_soa


# Function supplied in example
def addsub(x, y):
    return x + y, x - y


# Returning a homogeneous tuple
int32_duple = types.UniTuple(types.int32, 2)
signature = int32_duple(types.int32, types.int32)
ptx, resty = compile_ptx_soa(addsub, signature, device=True)
print(ptx)

# Returning a heterogeneous tuple
heterogeneous_duple = types.Tuple((types.int32, types.float32))
signature = heterogeneous_duple(types.int32, types.int32)
ptx, resty = compile_ptx_soa(addsub, signature, device=True)
print(ptx)


# Ensure that return of non-tuples still works
def unary_foo(x, y):
    return x + y


signature = types.int32(types.int32, types.int32)
ptx, resty = compile_ptx_soa(unary_foo, signature, device=True)
print(ptx)
