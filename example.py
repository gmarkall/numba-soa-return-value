from llvmlite import ir
from numba import types
from numba.core import sigutils
from numba.core.callconv import BaseCallConv
from numba.core.compiler_lock import global_compiler_lock
from numba.cuda.compiler import compile_cuda


class SoACallConv(BaseCallConv):
    """
    Calling convention aimed at matching the CUDA C/C++ ABI. The implemented
    function signature is:

        <Python return type> (<Python arguments>)

    Exceptions are unsupported in this convention
    """

    def _make_call_helper(self, builder):
        # Not needed for wrapping functions only.
        return None

    def return_value(self, builder, retval):
        return builder.ret(retval)

    def return_user_exc(self, builder, exc, exc_args=None, loc=None,
                        func_name=None):
        msg = "Python exceptions are unsupported in the CUDA C/C++ ABI"
        raise NotImplementedError(msg)

    def return_status_propagate(self, builder, status):
        msg = "Return status is unsupported in the CUDA C/C++ ABI"
        raise NotImplementedError(msg)
        pass

    def get_function_type(self, restype, argtypes):
        """
        Get the LLVM IR Function type for *restype* and *argtypes*.
        """
        arginfo = self._get_arg_packer(argtypes)
        argtypes = list(arginfo.argument_types)
        fnty = ir.FunctionType(self.get_return_type(restype), argtypes)
        return fnty

    def decorate_function(self, fn, args, fe_argtypes, noalias=False):
        """
        Set names and attributes of function arguments.
        """
        assert not noalias
        arginfo = self._get_arg_packer(fe_argtypes)
        arginfo.assign_names(self.get_arguments(fn),
                             ['arg.' + a for a in args])

    def get_arguments(self, func):
        """
        Get the Python-level arguments of LLVM *func*.
        """
        return func.args

    def call_function(self, builder, callee, resty, argtys, args):
        """
        Call the Numba-compiled *callee*.
        """
        arginfo = self._get_arg_packer(argtys)
        realargs = arginfo.as_arguments(builder, args)
        code = builder.call(callee, realargs)
        out = self.context.get_returned_value(builder, resty, code)
        return None, out

    def get_return_type(self, ty):
        return self.context.data_model_manager[ty].get_return_type()


def soa_wrap_function(context, lib, fndesc, nvvm_options):
    """
    Wrap a Numba ABI function such that it returns tuple values into SoA
    arguments.
    """
    device_function_name = lib.name
    library = lib.codegen.create_library(f'{lib.name}_function_',
                                         entry_name=device_function_name,
                                         nvvm_options=nvvm_options)
    library.add_linking_library(lib)

    # Determine the caller (C ABI) and wrapper (Numba ABI) function types
    argtypes = fndesc.argtypes
    restype = fndesc.restype
    c_call_conv = SoACallConv(context)
    wrapfnty = c_call_conv.get_function_type(restype, argtypes)
    fnty = context.call_conv.get_function_type(fndesc.restype, argtypes)

    # Create a new module and declare the callee
    wrapper_module = context.create_module("cuda.soa.wrapper")
    func = ir.Function(wrapper_module, fnty, fndesc.llvm_func_name)

    # Define the caller - populate it with a call to the callee and return
    # its return value

    wrapfn = ir.Function(wrapper_module, wrapfnty, device_function_name)
    builder = ir.IRBuilder(wrapfn.append_basic_block(''))

    arginfo = context.get_arg_packer(argtypes)
    callargs = arginfo.from_arguments(builder, wrapfn.args)
    # We get (status, return_value), but we ignore the status since we
    # can't propagate it through the C ABI anyway
    _, return_value = context.call_conv.call_function(
        builder, func, restype, argtypes, callargs)
    builder.ret(return_value)

    library.add_ir_module(wrapper_module)
    library.finalize()
    return library


@global_compiler_lock
def compile_ptx_soa(pyfunc, sig, debug=False, lineinfo=False, device=False,
                    fastmath=False, cc=None, opt=True):
    # This is just a copy of Numba's compile_ptx, with a modification to return
    # values as SoA and some simplifications to keep it short
    nvvm_options = {
        'fastmath': fastmath,
        'opt': 3 if opt else 0
    }

    args, return_type = sigutils.normalize_signature(sig)

    cc = cc or (5, 0)
    cres = compile_cuda(pyfunc, return_type, args, debug=debug,
                        lineinfo=lineinfo, fastmath=fastmath,
                        nvvm_options=nvvm_options, cc=cc)
    resty = cres.signature.return_type

    tgt = cres.target_context

    if device:
        lib = soa_wrap_function(tgt, cres.library, cres.fndesc, nvvm_options)
    else:
        code = pyfunc.__code__
        filename = code.co_filename
        linenum = code.co_firstlineno

        lib, kernel = tgt.prepare_cuda_kernel(cres.library, cres.fndesc, debug,
                                              lineinfo, nvvm_options, filename,
                                              linenum)

    ptx = lib.get_asm_str(cc=cc)
    return ptx, resty


def foo(x, y):
    return x + y, x - y


int32_duple = types.UniTuple(types.int32, 2)
signature = int32_duple(types.int32, types.int32)

ptx, resty = compile_ptx_soa(foo, signature, device=True)

print(ptx)
