from numba import types
from numba_soa import compile_ptx_soa


def foo(x, y):
    return x + y, x - y


int32_duple = types.UniTuple(types.int32, 2)
signature = int32_duple(types.int32, types.int32)

ptx, resty = compile_ptx_soa(foo, signature, device=True)

print(ptx)
