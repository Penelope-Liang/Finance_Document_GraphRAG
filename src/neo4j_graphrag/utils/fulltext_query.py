#  Copyright (c) "Neo4j"
#  Neo4j Sweden AB [https://neo4j.com]
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Sanitize user query text for Neo4j fulltext (Lucene) to avoid ParseException / TokenMgrError.
"""

import re


# Lucene query parser special characters (must be escaped with backslash)
# Order: backslash first, then two-char operators, then single chars
# `/` is included because Lucene 8+ treats it as the regex query delimiter.
_LUCENE_ESCAPE_PATTERN = re.compile(
    r"(\\|&&|\|\||[+\-!(){}\[\]^\"~*?:/])"
)


def sanitize_lucene_query(query: str | None) -> str:
    """
    Escape Lucene special characters in the query string so it can be passed
    to db.index.fulltext.queryNodes() without ParseException / TokenMgrError.

    Empty or whitespace-only query is replaced with "*" (match-all) so the
    fulltext branch still runs.

    Args:
        query: Raw query text (e.g. user question).

    Returns:
        Query safe for Lucene fulltext.
    """
    if query is None:
        return "*"
    s = query.strip()
    if not s:
        return "*"
    # Escape: prefix each special character/sequence with backslash
    escaped = _LUCENE_ESCAPE_PATTERN.sub(lambda m: "\\" + m.group(0), s)
    return escaped
