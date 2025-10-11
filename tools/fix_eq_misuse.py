import sys
import pathlib
import libcst as cst
import libcst.matchers as m

"""
Refactor rule:
- Replace `X.eq("literal")` or `X.eq('literal')` with `X == "literal"` WHEN X is a plain Python expr (Name/Attribute)
  and NOT obviously an Earth Engine object.
- Preserve legitimate Earth Engine uses:
   - ee.Image(...).eq(...)
   - ee.Number(...).eq(...)
   - Any call where the callee base looks like ee.<Type> or the receiver is a call to ee.<Type>(...)
- Also handle cases like: if method.eq("ndvi_kmeans"):  -> if method == "ndvi_kmeans":
Limitations:
- Heuristic-based; errs on the side of *not* changing when unsure.
"""

EE_CONSTRUCTOR_NAMES = {"Image", "Number", "Feature", "FeatureCollection", "Geometry", "Dictionary", "List", "Array"}


class EqMisuseTransformer(cst.CSTTransformer):
    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.BaseExpression:
        # We only care about Attribute-call like <receiver>.eq(...)
        if not isinstance(updated_node.func, cst.Attribute):
            return updated_node

        attr: cst.Attribute = updated_node.func
        if attr.attr.value != "eq":
            return updated_node

        # Must have exactly 1 positional arg which is a string literal to be a candidate
        if len(updated_node.args) != 1 or updated_node.args[0].keyword is not None:
            return updated_node

        arg0 = updated_node.args[0].value
        if not isinstance(arg0, (cst.SimpleString,)):
            # If eq is used with non-string (e.g., numbers, Images), leave it alone (likely EE use)
            return updated_node

        # Decide if the receiver is likely an EE object; if so, KEEP .eq(...)
        recv = attr.value

        # Pattern 1: ee.<Constructor>(...).eq(...)  -> keep
        if m.matches(
            recv,
            m.Call(
                func=m.Attribute(
                    value=m.Name("ee"),
                    attr=m.OneOf(*[m.Name(n) for n in EE_CONSTRUCTOR_NAMES]),
                )
            ),
        ):
            return updated_node

        # Pattern 2: ee.Image|Number|... attribute chain; e.g. ee.Image(x).select(...).eq(...)
        # If the base of the chain is a Call to ee.<Constructor>(...), keep
        base = recv
        ee_constructor_chain = False
        while isinstance(base, cst.Attribute):
            base = base.value
        if isinstance(base, cst.Call) and isinstance(base.func, cst.Attribute):
            if isinstance(base.func.value, cst.Name) and base.func.value.value == "ee" and isinstance(
                base.func.attr, cst.Name
            ):
                if base.func.attr.value in EE_CONSTRUCTOR_NAMES:
                    ee_constructor_chain = True
        if ee_constructor_chain:
            return updated_node

        # Pattern 3: ee.<something> direct (unlikely)
        if isinstance(recv, cst.Attribute) and isinstance(recv.value, cst.Name) and recv.value.value == "ee":
            # If it's ee.<something>.* and we can't prove, be conservative -> keep
            return updated_node

        # Otherwise, we assume Python-level misuse: replace X.eq("lit") -> X == "lit"
        left = recv
        right = arg0
        return cst.Comparison(left=left, comparisons=[cst.ComparisonTarget(operator=cst.Equal(), comparator=right)])


def process_file(path: pathlib.Path):
    src = path.read_text(encoding="utf-8")
    try:
        mod = cst.parse_module(src)
    except Exception as e:  # pragma: no cover - parsing failure is logged
        print(f"Skipping unparsable {path}: {e}")
        return
    new_mod = mod.visit(EqMisuseTransformer())
    if new_mod.code != src:
        path.write_text(new_mod.code, encoding="utf-8")
        print(f"Rewrote: {path}")


def main() -> int:
    root = pathlib.Path(".")
    for p in root.rglob("*.py"):
        # Skip virtual envs, build dirs, and this tool
        if any(part in {"venv", ".venv", "env", ".git", "__pycache__", "site-packages", "build", "dist"} for part in p.parts):
            continue
        if str(p).endswith("tools/fix_eq_misuse.py"):
            continue
        process_file(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
