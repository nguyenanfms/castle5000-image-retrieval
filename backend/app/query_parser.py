"""Conservative English clause extraction for local region search.

This is a deterministic splitter, not an LLM: preserve attributes and negations,
only split explicit new subjects; expose output for user correction.
"""
import re

_SUBJECT = r'(?:(?:a|an|the|another|other|two|three|some|several)\s+)?(?:man|woman|person|people|men|women|others|someone|somebody|boy|girl|guys)\b'
_BOUNDARY = re.compile(r'(?:\s+(?:and|while|whereas|but)\s+|,\s*)(?='+_SUBJECT+r')', re.I)


def split_query(query: str, limit: int = 5) -> list[str]:
    text = re.sub(r'\s+', ' ', query).strip()
    if not text:
        return []
    # Keep common title abbreviations together with their names.
    text = re.sub(r'\b(Mr|Mrs|Ms|Dr)\.', lambda m: m[1]+'<DOT>', text)
    sentences = re.split(r'(?<=[.!?;])\s+|\s*;\s*', text)
    conditions = []
    for sentence in sentences:
        for clause in _BOUNDARY.split(sentence):
            clause = clause.replace('<DOT>', '.').strip(' ,;.!?')
            if not clause:
                continue
            # A pronoun/attribute continuation belongs to the preceding subject.
            continuation = re.match(r'^(?:he|she|his|her|it|its|they|their|the sweater|the shirt|the upper|the lower)\b', clause, re.I)
            if continuation and conditions:
                conditions[-1] += '. ' + clause
            else:
                conditions.append(clause)
    unique = []
    for condition in conditions:
        if condition.casefold() not in {q.casefold() for q in unique}:
            unique.append(condition)
    # Keep excess clauses in the final slot, rather than silently dropping facts.
    if len(unique) > limit:
        unique = unique[:limit-1] + ['. '.join(unique[limit-1:])]
    return unique
