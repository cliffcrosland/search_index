# Search Index

A document search index with the following goals:
- High performance type-ahead search for 1M documents, each containing a few
  hundred words.
- Persistence to disk.

TODOs:
[x] Use spelling correction for queries to make interaction easy.
[x] Run simple HTTP JSON API server.
[x] Build simple HTML and JS testing interface.
[ ] Load large genealogy data set into index and interact with it in the UI.
[ ] Fix "single character" query term issue. Shouldn't have empty spelling correction suggestions.
[ ] Sort search results
  [ ] TF-IDF? How closely words appear together? Prioritize field?
[ ] Persist data to disk to recover quickly from closing, crashes.
