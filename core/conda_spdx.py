"""Conda-only SPDX cleanup. No general licence compatibility/intersection rules."""
import re


_VERSIONED_GNU = re.compile(r'^(A?GPL|LGPL)-(\d+(?:\.\d+)*)-(only|or-later)$', re.I)


def narrow_metadata_licenses(declarations):
    """Keep all original data outside this helper; narrow only exact standalone pairs."""
    unique = list(dict.fromkeys(declarations))
    only = {value.casefold() for value in unique if _VERSIONED_GNU.fullmatch(value) and value.lower().endswith('-only')}
    return [value for value in unique if not (
        _VERSIONED_GNU.fullmatch(value) and value.lower().endswith('-or-later')
        and (value[:-len('-or-later')] + '-only').casefold() in only
    )]


def reduce_only_or_later(expression):
    """Reduce equal-family/equal-version GNU only/later siblings in AND groups.

    OR branches and WITH exceptions remain intact. Different versions/families
    retain their SPDX expression, even when another compatibility rule exists.
    Invalid expressions raise ValueError instead of losing their structure.
    """
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError('Empty SPDX expression')
    tokens = re.findall(r'\(|\)|[^\s()]+', expression)
    position = 0

    def term():
        nonlocal position
        if position >= len(tokens):
            raise ValueError('Missing SPDX operand')
        value = tokens[position]
        if value == '(':
            position += 1
            result = operation('OR')
            if position >= len(tokens) or tokens[position] != ')':
                raise ValueError('Unbalanced SPDX parentheses')
            position += 1
            return result
        if value == ')' or value.upper() in {'AND','OR','WITH'} or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.+:\-]*', value):
            raise ValueError('Invalid SPDX identifier')
        position += 1
        if position < len(tokens) and tokens[position].upper() == 'WITH':
            position += 1
            if position >= len(tokens) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.\-]*', tokens[position]) or tokens[position].upper() in {'AND','OR','WITH'}:
                raise ValueError('Missing SPDX exception')
            value += ' WITH ' + tokens[position]
            position += 1
        return ('ID', value)

    def operation(operator):
        nonlocal position
        operand = term if operator == 'AND' else lambda: operation('AND')
        children = [operand()]
        while position < len(tokens) and tokens[position].upper() == operator:
            position += 1
            children.append(operand())
        flattened = []
        for child in children:
            flattened.extend(child[1] if child[0] == operator else [child])
        flattened = list(dict.fromkeys(flattened))
        if operator == 'AND':
            ids = [node[1] for node in flattened if node[0] == 'ID']
            retained = set(narrow_metadata_licenses(ids))
            flattened = [node for node in flattened if node[0] != 'ID' or node[1] in retained]
        return flattened[0] if len(flattened) == 1 else (operator, tuple(flattened))

    tree = operation('OR')
    if position != len(tokens):
        raise ValueError('Trailing SPDX tokens')

    def render(node, parent=None):
        kind, value = node
        if kind == 'ID':
            return value
        text = f' {kind} '.join(render(child, kind) for child in value)
        return f'({text})' if kind == 'OR' and parent == 'AND' else text
    return render(tree)
