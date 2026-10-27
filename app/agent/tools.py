"""
Agent tools (LangChain)
=======================
Concept: a *tool* is a function the agent can choose to call instead of (or
in addition to) searching documents. LangChain's @tool decorator wraps a
plain Python function so it has a name, a description, and a typed input
schema -- the same format LLM tool-calling APIs use.

The calculator is deliberately NOT built on eval(): it walks the parsed
syntax tree and only allows arithmetic nodes, so input like
"__import__('os').system(...)" is rejected instead of executed.
"""

import ast
import operator
import re
from typing import Optional

from langchain_core.tools import tool

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_eval(expression: str) -> float:
    """Evaluate a pure-arithmetic expression. Raises ValueError otherwise."""
    expression = expression.replace("^", "**")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Not a valid expression: {expression!r}") from e

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent too large")
            return _BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        raise ValueError("Only numbers and + - * / % ^ ( ) are allowed")

    return _eval(tree)


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as '12 * (3 + 4)'. Use for math questions."""
    try:
        result = safe_eval(expression)
    except ZeroDivisionError:
        return "Error: division by zero"
    except ValueError as e:
        return f"Error: {e}"
    return str(int(result)) if float(result).is_integer() else str(round(result, 6))


_EXPR_RE = re.compile(r"[\d\.\s\+\-\*/\^\(\)%]{3,}")


def extract_math_expression(question: str) -> Optional[str]:
    """
    Pull a candidate arithmetic expression out of a natural-language question,
    e.g. 'what is 15 * 4?' -> '15 * 4'. Returns None unless the candidate has
    a digit, an operator, and actually parses -- so '30-day returns' is NOT math.
    """
    q = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", " * ", question)
    candidates = [m.group().strip() for m in _EXPR_RE.finditer(q)]
    for cand in sorted(candidates, key=len, reverse=True):
        if re.search(r"\d", cand) and re.search(r"[\+\-\*/\^%]", cand):
            try:
                safe_eval(cand)
                return cand
            except (ValueError, ZeroDivisionError):
                continue
    return None
