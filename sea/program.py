"""Abstraction programs: LLM-written ``phi(s)`` functions over privileged state.

A program is Python source defining ``def phi(s): ... return {name: tensor}``. ``s`` maps
privileged field names to torch tensors of shape (..., dim); the program must be written
with broadcasting tensor operations so that it runs unchanged on a single state, on a
(B, T) training batch on the GPU, and on a whole replay dataset.

Source is checked against an AST whitelist before it is executed. This keeps honest
LLM mistakes (imports, file access, unbounded loops) out; it is not a security boundary
against a deliberately adversarial author.
"""

import ast
import hashlib
import math

import torch

MAX_SOURCE_CHARS = 6000  # LLM-written programs; generated baselines may pass a larger limit
MAX_OUTPUT_DIM = 32

_ALLOWED_NODES = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign, ast.AugAssign, ast.Expr,
    ast.If, ast.IfExp, ast.For, ast.Pass, ast.Name, ast.Load, ast.Store, ast.Constant, ast.Tuple, ast.List,
    ast.Dict, ast.Subscript, ast.Slice, ast.Attribute, ast.Call, ast.keyword, ast.BinOp, ast.UnaryOp,
    ast.BoolOp, ast.Compare, ast.ListComp, ast.DictComp, ast.GeneratorExp, ast.comprehension, ast.Starred,
    ast.operator, ast.unaryop, ast.boolop, ast.cmpop,
)  # fmt: skip
_FORBIDDEN_NAMES = {
    "eval", "exec", "open", "compile", "getattr", "setattr", "delattr", "globals", "locals", "vars",
    "input", "breakpoint", "help", "type", "object", "super", "memoryview", "classmethod", "staticmethod",
}  # fmt: skip
# Tensor methods that touch the filesystem, other processes or autograd internals.
_FORBIDDEN_ATTRS = {
    "numpy", "tolist", "item", "share_memory_", "register_hook", "backward", "storage", "untyped_storage",
    "data_ptr", "cuda", "cpu", "to", "requires_grad_", "pin_memory", "save", "load",
}  # fmt: skip
_SAFE_BUILTINS = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in ("len", "range", "min", "max", "abs", "float", "int", "bool", "enumerate", "zip", "sum", "tuple", "list", "dict", "sorted", "reversed", "round")  # fmt: skip  # noqa: E501
}


class ProgramError(Exception):
    pass


# ---- helper library exposed to programs -------------------------------------------------
def norm(x, eps=1e-8):
    """Euclidean norm over the last axis, keeping it: (..., k) -> (..., 1)."""
    return torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


def dist(a, b):
    """Distance between two points: (..., k), (..., k) -> (..., 1)."""
    return norm(a - b)


def unit(x, eps=1e-6):
    return x / norm(x).clamp_min(eps)


def dot(a, b):
    return (a * b).sum(-1, keepdim=True)


def cos_between(a, b):
    """Cosine of the angle between two vectors, in [-1, 1]: -> (..., 1)."""
    return dot(unit(a), unit(b)).clamp(-1.0, 1.0)


def cross(a, b):
    return torch.linalg.cross(a, b, dim=-1)


def quat_rotate(q, v):
    """Rotate vector v (..., 3) by quaternion q (..., 4) given as (w, x, y, z)."""
    q = unit(q)
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * torch.linalg.cross(xyz, v, dim=-1)
    return v + w * t + torch.linalg.cross(xyz, t, dim=-1)


def quat_axis(q, axis=2):
    """World direction of a body's local x/y/z axis (axis = 0/1/2): (..., 4) -> (..., 3)."""
    e = torch.zeros(q.shape[:-1] + (3,), dtype=q.dtype, device=q.device)
    e[..., axis] = 1.0
    return quat_rotate(q, e)


def planar(x):
    """Horizontal (x, y) components: (..., 3) -> (..., 2)."""
    return x[..., :2]


def height(x):
    """Vertical (z) component: (..., 3) -> (..., 1)."""
    return x[..., 2:3]


def soft_gate(x, threshold, sharpness=50.0):
    """Smooth indicator of x > threshold, in (0, 1)."""
    return torch.sigmoid(sharpness * (x - threshold))


def linear(x, weights):
    """x (..., k) times a constant matrix given as nested lists [out][k]; used by generated baselines."""
    w = torch.tensor(weights, dtype=x.dtype, device=x.device)
    return x @ w.T


HELPERS = {
    f.__name__: f
    for f in (norm, dist, unit, dot, cos_between, cross, quat_rotate, quat_axis, planar, height, soft_gate, linear)
}
_TORCH_FUNCS = (
    "abs", "sqrt", "exp", "log", "log1p", "tanh", "sigmoid", "sin", "cos", "atan2", "acos", "asin", "sign",
    "clamp", "minimum", "maximum", "where", "cat", "stack", "zeros_like", "ones_like", "relu", "square",
    "isfinite", "logical_and", "logical_or", "logical_not",
)  # fmt: skip
TORCH_NAMESPACE = {name: getattr(torch, name) for name in _TORCH_FUNCS}

HELPER_DOC = "\n".join(
    f"- {name}{_sig}: {(fn.__doc__ or '').strip()}"
    for name, fn, _sig in (
        ("norm", norm, "(x)"),
        ("dist", dist, "(a, b)"),
        ("unit", unit, "(x)"),
        ("dot", dot, "(a, b)"),
        ("cos_between", cos_between, "(a, b)"),
        ("cross", cross, "(a, b)"),
        ("quat_rotate", quat_rotate, "(q, v)"),
        ("quat_axis", quat_axis, "(q, axis=2)"),
        ("planar", planar, "(x)"),
        ("height", height, "(x)"),
        ("soft_gate", soft_gate, "(x, threshold, sharpness=50.0)"),
    )
)


def _check_ast(tree, extra_names=()):
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(funcs) != len(tree.body) or not funcs:
        raise ProgramError("source must contain only function definitions")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ProgramError(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and (node.id in _FORBIDDEN_NAMES or node.id.startswith("__")):
            raise ProgramError(f"disallowed name: {node.id}")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in _FORBIDDEN_ATTRS):
            raise ProgramError(f"disallowed attribute: {node.attr}")
        if isinstance(node, ast.FunctionDef) and (node.decorator_list or node.name.startswith("_")):
            raise ProgramError("decorators and private function names are not allowed")
        if isinstance(node, ast.Constant) and isinstance(node.value, (bytes, complex)):
            raise ProgramError("disallowed constant")
    return funcs


def compile_functions(source, namespace, max_chars=MAX_SOURCE_CHARS):
    """Validate ``source`` and execute it; returns {function name: callable}."""
    if len(source) > max_chars:
        raise ProgramError(f"source longer than {max_chars} characters")
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise ProgramError(f"syntax error: {e}") from e
    funcs = _check_ast(tree)
    scope = {"__builtins__": _SAFE_BUILTINS, "math": math, **TORCH_NAMESPACE, **HELPERS, **namespace}
    exec(compile(tree, "<abstraction>", "exec"), scope)  # noqa: S102 - AST-checked above
    return {f.name: scope[f.name] for f in funcs}


def fields_used(source):
    """Privileged field names a program reads, found as string subscripts of ``s``."""
    used = []
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "s"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
            and node.slice.value not in used
        ):
            used.append(node.slice.value)
    return used


def source_size(source):
    """Number of AST nodes: a formatting-independent measure of program length."""
    return sum(1 for _ in ast.walk(ast.parse(source)))


class Program:
    """A validated abstraction program bound to a privileged-state layout."""

    def __init__(self, source, spec, helpers=None, name=None, max_dim=MAX_OUTPUT_DIM):
        self.source = source.strip() + "\n"
        self.spec = spec
        self.helpers = dict(helpers or {})  # {helper name: source}, shared library code
        self.pid = hashlib.sha1((self.source + "".join(sorted(self.helpers.values()))).encode()).hexdigest()[:12]
        self.name = name or self.pid
        namespace = {}
        for helper_source in self.helpers.values():
            namespace.update(compile_functions(helper_source, namespace))
        funcs = compile_functions(self.source, namespace, max_chars=max(MAX_SOURCE_CHARS, 400 * max_dim))
        if "phi" not in funcs:
            raise ProgramError("source must define `def phi(s):`")
        self._fn = funcs["phi"]
        self.fields = fields_used(self.source)
        missing = [f for f in self.fields if f not in spec.slices]
        if missing:
            raise ProgramError(f"unknown privileged fields: {missing}")
        self.size = source_size(self.source)
        # Probe once on zeros to fix the output layout; values are checked on real data later.
        probe = self._run(torch.zeros(2, 3, spec.dim))
        self.feature_names = list(probe)
        self.feature_dims = [int(v.shape[-1]) for v in probe.values()]
        self.dim = sum(self.feature_dims)
        if not 1 <= self.dim <= max_dim:
            raise ProgramError(f"total output dimension {self.dim} outside [1, {max_dim}]")

    @property
    def dim_names(self):
        out = []
        for name, d in zip(self.feature_names, self.feature_dims):
            out += [name] if d == 1 else [f"{name}[{i}]" for i in range(d)]
        return out

    def _run(self, priv):
        lead = priv.shape[:-1]
        try:
            out = self._fn(self.spec.split(priv.float()))
        except ProgramError:
            raise
        except Exception as e:  # noqa: BLE001 - surfaced to the LLM as feedback
            raise ProgramError(f"runtime error: {type(e).__name__}: {e}") from e
        if not isinstance(out, dict) or not out:
            raise ProgramError("phi must return a non-empty dict {feature name: tensor}")
        clean = {}
        for key, value in out.items():
            if not isinstance(key, str) or not key.isidentifier():
                raise ProgramError(f"feature name {key!r} is not a valid identifier")
            if not isinstance(value, torch.Tensor):
                raise ProgramError(f"feature {key!r} is not a tensor")
            value = value.float()
            if value.shape == lead:
                value = value.unsqueeze(-1)
            if value.shape[:-1] != lead or value.dim() != len(lead) + 1:
                raise ProgramError(
                    f"feature {key!r} has shape {tuple(value.shape)}; expected leading dims {tuple(lead)} "
                    "plus one feature axis (use keepdim-style helpers, do not reduce over batch/time)"
                )
            clean[key] = value
        return clean

    @torch.no_grad()
    def __call__(self, priv):
        """(..., priv_dim) -> (..., self.dim). Non-finite entries are left for the caller to mask."""
        out = self._run(priv)
        if list(out) != self.feature_names or [int(v.shape[-1]) for v in out.values()] != self.feature_dims:
            raise ProgramError("output layout depends on the input values")
        return torch.cat(list(out.values()), dim=-1)

    def to_json(self):
        return {"name": self.name, "pid": self.pid, "source": self.source, "helpers": self.helpers}


def identity_source(spec, fields=None):
    """Program that returns raw privileged fields (the 'predict the raw state' baseline)."""
    names = list(fields or spec.names)
    body = ", ".join(f'"{_ident(n)}": s["{n}"]' for n in names)
    return f"def phi(s):\n    return {{{body}}}\n"


def random_projection_source(spec, out_dim, seed=0, fields=None):
    """Program returning `out_dim` fixed random linear mixtures of the raw privileged state.

    A control for 'does the *choice* of abstraction matter': same information source and
    target count as an evolved abstraction, but no structure.
    """
    names = list(fields or spec.names)
    total = sum(spec.slices[n].stop - spec.slices[n].start for n in names)
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(out_dim, total, generator=g) / total**0.5).tolist()
    w = [[round(v, 4) for v in row] for row in w]
    parts = ", ".join(f's["{n}"]' for n in names)
    return f'def phi(s):\n    x = cat([{parts}], -1)\n    return {{"random_mix": linear(x, {w})}}\n'


def _ident(name):
    out = "".join(c if c.isalnum() else "_" for c in name)
    return out if out[0].isalpha() else "f_" + out
