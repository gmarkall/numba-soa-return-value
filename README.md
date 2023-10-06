# Numba SoA interface example

A demonstration of returning values into an SoA data structure from a
Numba-compiled Python function.

## Examples

The Python function:

```python
def addsub(x, y):
    return x + y, x - y
```

compiled with the `compile_ptx_soa()` function where:

- Arguments are of `int32` type
- The return type is a tuple of `int32`

compiles to PTX equivalent to th C function:

```C
void (int32_t *r1, int32_t *r2, int32_t x, int32_t y)
{
  *r1 = x + y;
  *r2 = x - y;
}
```

Or, as the actual PTX produced:

```asm
.visible .func addsub(
	.param .b64 addsub_param_0,
	.param .b64 addsub_param_1,
	.param .b32 addsub_param_2,
	.param .b32 addsub_param_3
)
{
	.reg .b32 	%r<5>;
	.reg .b64 	%rd<3>;


	ld.param.u64 	%rd1, [addsub_param_0];
	ld.param.u64 	%rd2, [addsub_param_1];
	ld.param.u32 	%r1, [addsub_param_2];
	ld.param.u32 	%r2, [addsub_param_3];
	add.s32 	%r3, %r2, %r1;
	sub.s32 	%r4, %r1, %r2;
	st.u32 	[%rd1], %r3;
	st.u32 	[%rd2], %r4;
	ret;

}
```

Returning a heterogeneous tuple is also possible, For example where the return
type is specified as a tuple of `(int32, float32)` we get:

```asm
.visible .func addsub(
	.param .b64 addsub_param_0,
	.param .b64 addsub_param_1,
	.param .b32 addsub_param_2,
	.param .b32 addsub_param_3
)
{
	.reg .f32 	%f<2>;
	.reg .b32 	%r<4>;
	.reg .b64 	%rd<6>;


	ld.param.u64 	%rd1, [addsub_param_0];
	ld.param.u64 	%rd2, [addsub_param_1];
	ld.param.u32 	%r1, [addsub_param_2];
	ld.param.u32 	%r2, [addsub_param_3];
	add.s32 	%r3, %r2, %r1;
	cvt.s64.s32 	%rd3, %r1;
	cvt.s64.s32 	%rd4, %r2;
	sub.s64 	%rd5, %rd3, %rd4;
	cvt.rn.f32.s64 	%f1, %rd5;
	st.u32 	[%rd1], %r3;
	st.f32 	[%rd2], %f1;
	ret;
}
```

(Note the `st.u32` for the first return value vs. `st.f32` for the second).

## Running the examples

Run:

```
python example.py
```

with a recent version of Numba (tested with 0.58, but probably 0.57 would
suffice).

This example generates and prints PTX for the two examples above, and also the
PTX for an example where a non-tuple type is returned. In the non-tuple case the
return value is placed in the location pointed to by the first parameter
(consistent with the "usual" Numba ABI. It could also be thought of as returning
into an array rather than a struct of arrays, if the caller passes a pointer to
the array element to be returned into.

