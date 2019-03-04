# Search Index

A document search index with the following goals:
- High performance type-ahead search for 1M documents, each containing a few
  hundred words.
- Persistence to disk.

TODOs:
[x] Use spelling correction for queries to make interaction easy.
[ ] Run simple HTTP JSON API server.
[ ] Sort search results.
  [ ] TF-IDF? How closely words appear together? Prioritize field?
[ ] Persist data to disk to recover quickly from closing, crashes.
