mod analyzer;

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::{HashMap, HashSet};

use analyzer::{Analyzer, Term};
use crate::skip_list::{Key, SkipList};

/// # Search Index
///
/// The search index consists of three main components:
///
/// - Term to Postings index: Finds documents that match words.
/// - 3-Gram to Terms index: Handles spelling corrections.
/// - Soundex to Terms index: Matches person names based on how they sound.
///
/// ## Definitions
///
/// ### Term
///
/// A string representing a normalized token found in a document. Example:
///
/// ```
/// tokens("Although it was cold, the view from Mt. Saint-Michel was beautiful!") =>
/// [
///     "although", "it", "was", "cold", "the", "view", "from", "mt", "saint",
///     "michel", "was", "beautiful"
/// ]
/// ```
///
/// ### Posting
///
/// Represents a word's location in a particular document. Example:
///
/// ```
/// documents = [
///     {
///         "id": 123,
///         "name": "Jane Smith",
///         "bio": "Hailing from SF, Jane Smith works at..."
///     },
///     {
///
///         "id": 234,
///         "name": "Beth Chou",
///         "bio": "Worked on a team with Jane Gomez...",
///     }
/// ]
///
/// postings("jane", documents) => [
///     { document_id: 123, field: "name", index: 0 },
///     { document_id: 123, field: "bio", index: 3 },
///     { document_id: 234, field: "bio", index: 5 },
/// ]
/// ```
///
/// ### 3-Gram
///
/// Three consecutive chars that appear in a term. Mapping from 3-Grams to terms
/// helps us handle spelling corrections. This can also help us handle wildcard
/// queries. Note that we also include a special "$" character to denote the
/// beginning or end of the word. Example:
///
/// ```
/// three_grams("benevolent") => [
///   "$be",  "ben", "ene", "nev", "evo", "vol", "ole", "len", "ent", "nt$"
/// ]
/// ```
///
/// ### Soundex
///
/// A four char string representing roughly how a term sounds. Particularly
/// useful for matching person names. The algorithm can be found in
/// 3.4 in "Chris Manning, Introduction to Information Retrieval, 1st Edition
/// (Cambridge University Press, July 7, 2008)". Example:
///
/// ```
/// soundex("herman") => "h655"
/// ```
///
/// The general idea is to assign letters that sound like one another to the
/// same digit. The algorithm for computing English name soundexes is as
/// follows:
/// 1. Keep the first letter of the term.
/// 2. Change these letters to 0: A, E, I, O, U, H, W, Y
/// 3. Change these letters to the following digits:
///    - B, F, P, V => 1
///    - C, G, J, K, Q, S, X, Z => 2
///    - D, T => 3
///    - L => 4
///    - M, N => 5
///    - R => 6
/// 4. Repeatedly coalesce repeated digits into one digit.
/// 5. Remove all zeroes from the result. If less than 4 chars long, pad the end
///    with zeroes.
///
/// ## Supported operations
///
/// - New(Config): Create a new Search Index with the given configuration.
///
/// - Index(DocumentId, DocumentJson): Insert the given document into the index.
///   Field data will be extracted according to the configuration passed into
///   Config. For example, if the fields we are capturing are `["name",
///   "location", "bio"]`, we will extract these fields from the document JSON
///   object: `DocumentJson["name"], Document["location"], Document["bio"]`.
///
/// - Remove(DocumentId): Delete the given document from the index.
///
/// - Search(Query) -> Hits: Given a string query, returns the top results
///   that match. The string query is broken up into normalized terms. We find
///   the documents that match each term, and we favor results where terms are
///   found:
///   - near each other in order
///   - in a favored field (i.e. low field_index)
///   - close to the front of that field (i.e. low term_index)
///
pub struct SearchIndex {
    // Configuration for this search index.
    config: Config,

    // We use skip-lists below to compute intersections efficiently.

    // Term to Postings index
    term_postings: HashMap<String, SkipList<Posting, ()>>,

    // 3-Gram to Terms index
    trigram_terms: HashMap<String, HashSet<String>>,

    // Soundex to Terms index
    soundex_terms: HashMap<String, HashSet<String>>,

    // The following maps are useful for supporting the remove(document_id)
    // operation. Can quickly look up the terms and document postings to remove.

    // Map from document_id to terms found in the document.
    id_terms: HashMap<u64, HashSet<String>>,

    // Map from document_id to Posting structs.
    id_postings: HashMap<u64, HashSet<Posting>>,

    // Map from term to document_ids. Useful for computing df(term) == num docs that contain term.
    term_ids: HashMap<String, HashSet<u64>>,

    // Map from (document_id, field_index) to original document text.
    //
    // Eg. Let document_id = 123, fields = ["name", "location", "bio"]
    //
    // If location value is "San Francisco", then text[(123, 1)] == "San Francisco".
    id_field_text: HashMap<(u64, usize), String>,

    // Map from (document_id, field_index) to normalized terms and where they were found in the
    // original text.
    //
    // Eg. let document_id = 123, fields = ["name", "location", "bio"]
    //
    // If location value is "San Francisco, CA 94115", then:
    //
    // text[(123, 1)] == [(0, 3, "san"), (5, 9, "francisco"), (18, 2, "ca"), (21, 5, "94115")]
    id_field_terms: HashMap<(u64, usize), Vec<Term>>,
}

#[derive(Clone, Debug)]
pub struct Config {
    // Which fields to extract from the document JSON objects to be stored in
    // this index. The fields will be favored in the order they appear in this
    // vector. For example, if documents contain `bio`, `name`, and `location`
    // fields, and we want to favor `name` hits above `location` above `bio`,
    // then this vector should be `["name", "location", "bio"]`.
    pub fields: Vec<String>,
}

// Represents the location of a term in a document.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct Posting {
    // The ID of the document.
    document_id: u64,

    // Which field the term was found in.
    field_index: usize,

    // Where the term was found within the field. For example, if the field is:
    // "Welcome to San Francisco", and the term is "san", then term_index
    // would be 2, since "san" appears at index 2 in ["welcome", "to", "san",
    // "francisco"].
    term_index: usize,
}

impl Key for Posting {
    fn key_cmp(&self, query: &Posting, prefix_match: bool) -> Ordering {
        if prefix_match {
            // Prefix match simply compares document_id
            return self.document_id.cmp(&query.document_id);
        }
        // Sorted by <document_id, field_index, term_index>
        let mut ord = self.document_id.cmp(&query.document_id);
        if ord != Ordering::Equal {
            return ord;
        }
        ord = self.field_index.cmp(&query.field_index);
        if ord != Ordering::Equal {
            return ord;
        }
        self.term_index.cmp(&query.term_index)
    }
}

#[derive(Serialize)]
pub struct SearchResult {
    pub spelling_correction: bool,
    pub query_terms: Vec<String>,
    pub hits: Vec<Hit>,
}

impl SearchResult {
    fn empty(query_terms: Vec<String>) -> SearchResult {
        SearchResult {
            query_terms,
            spelling_correction: false,
            hits: Vec::new(),
        }
    }
}

// Represents a search result hit.
#[derive(Clone, Debug, Serialize)]
pub struct Hit {
    // The document where the search result was found
    pub document_id: u64,

    // Pairs of (field_index, field_name) that matched query
    pub fields: HashMap<usize, String>,

    // Map from field_index (eg. "name", "bio", etc.) to snippets that show nearby text in source
    // document.
    pub field_snippets: HashMap<usize, Snippet>,
}

impl Hit {
    fn new(document_id: u64) -> Hit {
        Hit {
            document_id,
            fields: HashMap::new(),
            field_snippets: HashMap::new(),
        }
    }
}

// Data structure that can be used to render snippet of nearby search text
#[derive(Clone, Debug, Serialize)]
pub struct Snippet {
    term_indices: Vec<usize>,
    pub body: String,
}

struct Candidate<'a> {
    query_term: String,
    postings: &'a SkipList<Posting, ()>,
}

impl<'a> Candidate<'a> {
    fn new(query_term: String, postings: &'a SkipList<Posting, ()>) -> Candidate<'a> {
        Candidate {
            query_term,
            postings,
        }
    }
}

struct CandidateIntersection {
    query_terms: Vec<String>,
    intersection: SkipList<Posting, ()>,
}

impl SearchIndex {
    pub fn new(config: Config) -> SearchIndex {
        SearchIndex {
            config,
            term_postings: HashMap::new(),
            trigram_terms: HashMap::new(),
            soundex_terms: HashMap::new(),
            id_terms: HashMap::new(),
            id_postings: HashMap::new(),
            term_ids: HashMap::new(),
            id_field_text: HashMap::new(),
            id_field_terms: HashMap::new(),
        }
    }

    pub fn index(&mut self, document_id: u64, document_json: &serde_json::Value) {
        for field_index in 0..self.config.fields.len() {
            let field = self.config.fields[field_index].clone();
            let text = Analyzer::field_to_text(document_json, &field);
            self.id_field_text
                .insert((document_id, field_index), text.clone());
            let terms = Analyzer::text_to_terms(&text);
            let terms_without_normalized_strs = terms
                .iter()
                .map(|term| Term {
                    text_first: term.text_first,
                    text_last: term.text_last,
                    term_index: term.term_index,
                    normalized: String::new(),
                })
                .collect();
            self.id_field_terms
                .insert((document_id, field_index), terms_without_normalized_strs);
            for (term_index, term) in terms.iter().enumerate() {
                let posting = Posting {
                    document_id,
                    field_index,
                    term_index,
                };
                self.insert_posting(&term.normalized, posting);
            }
        }
    }

    pub fn search(&self, query: &str) -> SearchResult {
        // Analyze query. Translate to terms.
        let query_terms = Analyzer::text_to_terms(query);
        let query_terms: Vec<String> = query_terms.into_iter().map(|t| t.normalized).collect();

        // Look up postings for each query term. If a query term wasn't found, use spelling
        // correction to find good alternative candidates.
        let mut query_term_candidates = Vec::with_capacity(query_terms.len());
        let query_terms_len = query_terms.len();

        let mut spelling_correction = false;
        let mut failure = false;
        for (query_term_idx, query_term) in query_terms.iter().enumerate() {
            if let Some(postings) = self.term_postings.get(query_term) {
                query_term_candidates.push(vec![Candidate::new(query_term.to_string(), postings)]);
            } else {
                // Look for spelling correction candidates.
                spelling_correction = true;
                let last_query_term = query_term_idx == query_terms_len - 1;
                let corrections = self.recommend_spelling_corrections(query_term, last_query_term);
                let mut candidates = Vec::new();
                for correction in corrections {
                    if let Some(postings) = self.term_postings.get(&correction) {
                        candidates.push(Candidate::new(correction, postings));
                    }
                }
                // If there are no candidate replacements, there are no documents matching all
                // of the query terms.
                if candidates.is_empty() {
                    failure = true;
                    break;
                }
                query_term_candidates.push(candidates);
            }
        }

        if failure {
            return SearchResult::empty(query_terms);
        }

        // Find document_id intersection of all postings.
        let candidate_intersection =
            Self::compute_candidate_intersection(&query_term_candidates, spelling_correction);
        if candidate_intersection.is_none() {
            return SearchResult::empty(query_terms);
        }
        let candidate_intersection = candidate_intersection.unwrap();

        //  Given postings, compute hits, summarizing search results for each document. Each hit
        //  shows field that matched and snippet showing surrounding text from original document.
        let mut hits: Vec<Hit> = Vec::new();

        for entry in candidate_intersection.intersection.iter() {
            let posting = &entry.key;
            let mut len = hits.len();
            if len == 0 || hits[len - 1].document_id != posting.document_id {
                let hit = Hit::new(posting.document_id);
                hits.push(hit);
                len += 1;
            }
            self.add_posting_to_hit(posting, &mut hits[len - 1]);
        }

        // Now that all hits are gathered, compute snippet text.
        for hit in hits.iter_mut() {
            for (field_index, ref mut snippet) in hit.field_snippets.iter_mut() {
                snippet.body =
                    self.compute_snippet_body(hit.document_id, *field_index, &snippet.term_indices);
            }
        }

        SearchResult {
            query_terms: candidate_intersection.query_terms,
            spelling_correction,
            hits,
        }
    }

    pub fn remove(&mut self, document_id: u64) {
        self.remove_term_postings(document_id);
        self.id_terms.remove(&document_id);
        self.id_postings.remove(&document_id);
    }

    fn insert_posting(&mut self, term: &str, posting: Posting) {
        // Update Term to Postings index
        let postings = self
            .term_postings
            .entry(term.to_string())
            .or_insert_with(SkipList::new);
        postings.set(&posting, ());

        // Update Trigram to Terms index
        for trigram in Analyzer::trigrams(term) {
            let terms = self
                .trigram_terms
                .entry(trigram)
                .or_insert_with(HashSet::new);
            terms.insert(term.to_string());
        }

        // Update Soundex to Terms index
        let soundex = Analyzer::soundex(term);
        let terms = self
            .soundex_terms
            .entry(soundex)
            .or_insert_with(HashSet::new);
        terms.insert(term.to_string());

        let document_id = posting.document_id;

        // Insert into id_terms
        let terms = self
            .id_terms
            .entry(document_id)
            .or_insert_with(HashSet::new);
        terms.insert(term.to_string());

        // Insert into id_postings
        let postings = self
            .id_postings
            .entry(document_id)
            .or_insert_with(HashSet::new);
        postings.insert(posting);

        // Insert into term_ids
        let document_ids = self
            .term_ids
            .entry(term.to_string())
            .or_insert_with(HashSet::new);
        document_ids.insert(document_id);
    }

    fn remove_term_postings(&mut self, document_id: u64) {
        let terms = self.id_terms.get(&document_id);
        if terms.is_none() {
            return;
        }
        let terms = terms.unwrap();
        let postings = self.id_postings.get(&document_id);
        if postings.is_none() {
            return;
        }
        let postings = postings.unwrap();
        for term in terms.iter() {
            let empty;
            {
                let term_postings = self.term_postings.get_mut(term);
                if term_postings.is_none() {
                    continue;
                }
                let term_postings = term_postings.unwrap();
                for posting in postings.iter() {
                    term_postings.remove(&posting);
                }
                empty = term_postings.is_empty();
            }
            if empty {
                self.term_postings.remove(term);
            }
        }
    }

    fn recommend_spelling_corrections(&self, term: &str, prefix_ok: bool) -> Vec<String> {
        const MAX_EDIT_DISTANCE: usize = 10;
        const NUM_CANDIDATES: usize = 10;

        // Compute 3-character trigrams of the term. Find indexed terms that have the same
        // trigrams. Compute the edit distance betwteen the term and each of the indexed terms. Use
        // a binary heap to select the top few terms with smallest edit distance.

        // Binary heap element definitions
        struct TermEditDist {
            term: String,
            edit_distance: usize,
        };
        impl PartialEq for TermEditDist {
            fn eq(&self, other: &Self) -> bool {
                other.edit_distance == self.edit_distance && other.term.len() == self.term.len()
            }
        }
        impl Eq for TermEditDist {}
        impl Ord for TermEditDist {
            fn cmp(&self, other: &Self) -> Ordering {
                other
                    .edit_distance
                    .cmp(&self.edit_distance)
                    .then_with(|| other.term.len().cmp(&self.term.len()))
            }
        };
        impl PartialOrd for TermEditDist {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        };
        impl TermEditDist {
            fn new(term: &str, edit_distance: usize) -> TermEditDist {
                TermEditDist {
                    term: term.to_string(),
                    edit_distance,
                }
            }
        }

        // First, look for good trigram match by edit distance. If `prefix_ok` flag is set, then we
        // consider `edit_distance` to be 0 if `term` is a prefix of the trigram match. Helps us
        // find matches as a user types.
        let trigrams = Analyzer::trigrams(term);
        let mut possible_corrections = HashSet::new();
        let mut edit_distances = Vec::new();
        for trigram in trigrams {
            if let Some(trigram_terms) = self.trigram_terms.get(&trigram) {
                for trigram_term in trigram_terms {
                    if possible_corrections.contains(trigram_term) {
                        continue;
                    }
                    possible_corrections.insert(trigram_term);
                    if prefix_ok && trigram_term.starts_with(term) {
                        edit_distances.push(TermEditDist::new(trigram_term, 0));
                        continue;
                    }
                    let edit_distance = Analyzer::edit_distance(trigram_term, term);
                    if edit_distance <= MAX_EDIT_DISTANCE {
                        edit_distances.push(TermEditDist::new(trigram_term, edit_distance));
                    }
                }
            }
        }

        // Take top few terms by edit distance
        let mut heap = BinaryHeap::from(edit_distances);
        let count = std::cmp::min(NUM_CANDIDATES, heap.len());
        let mut ret = Vec::new();
        for _ in 0..count {
            ret.push(heap.pop().unwrap().term);
        }
        ret
    }

    fn compute_candidate_intersection(
        query_term_candidates: &[Vec<Candidate>],
        spelling_correction: bool,
    ) -> Option<CandidateIntersection> {
        // If spelling correction was not needed, there is one candidate per query term.
        if !spelling_correction {
            let candidates: Vec<&Candidate> =
                query_term_candidates.iter().map(|qtc| &qtc[0]).collect();
            let candidate_postings: Vec<&SkipList<Posting, ()>> =
                candidates.iter().map(|c| c.postings).collect();
            let query_terms: Vec<String> =
                candidates.iter().map(|c| c.query_term.clone()).collect();
            let intersection = Self::find_document_intersection(candidate_postings);
            return Some(CandidateIntersection {
                query_terms,
                intersection,
            });
        }

        // If spelling correction was needed, there may be multiple candidates per query term. Try
        // a few selections of candidates using the following algorithm:
        //
        // 1. Take the current candidate for each query term (starting at index 0).
        // 2. Find the document intersection of these candidates.
        // 3. If the intersection is non-zero, return it.
        // 4. Otherwise, use the heap to find the query term that has the smallest postings size.
        //    That might be the query term that is limiting our ability to find an intersection
        //    because its posting list is small (just a heuristic). Select the next candidate for
        //    that query term, and repeat at step 1. If we have exhausted the candidates for any
        //    query term, then exit since no intersection can be found.
        const MAX_ATTEMPTS: usize = 10;

        // Binary heap element definitions
        struct CandidateIter<'a> {
            query_term_index: usize,
            candidates: &'a Vec<Candidate<'a>>,
            i: usize,
        };
        impl<'a> PartialEq for CandidateIter<'a> {
            fn eq(&self, other: &Self) -> bool {
                self.query_term_index == other.query_term_index && self.i == other.i
            }
        }
        impl<'a> Eq for CandidateIter<'a> {}
        impl<'a> Ord for CandidateIter<'a> {
            fn cmp(&self, other: &Self) -> Ordering {
                other.candidates[other.i]
                    .postings
                    .len()
                    .cmp(&self.candidates[self.i].postings.len())
            }
        };
        impl<'a> PartialOrd for CandidateIter<'a> {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        // Initialize heap
        let iters: Vec<CandidateIter> = query_term_candidates
            .iter()
            .enumerate()
            .map(|(query_term_index, candidates)| CandidateIter {
                query_term_index,
                candidates: &candidates,
                i: 0,
            })
            .collect();
        let mut heap = BinaryHeap::from(iters);

        // Make a finite number of attempts to find an intersection
        for _ in 0..MAX_ATTEMPTS {
            let mut candidates: Vec<(usize, &Candidate)> = heap
                .iter()
                .map(|ci| (ci.query_term_index, &ci.candidates[ci.i]))
                .collect();
            let candidate_postings: Vec<&SkipList<Posting, ()>> = candidates
                .iter()
                .map(|&(_query_term_index, candidate)| candidate.postings)
                .collect();
            let intersection = Self::find_document_intersection(candidate_postings);

            // If intersection found, return it.
            if !intersection.is_empty() {
                candidates.sort_by_key(|&(query_term_index, _candidate)| query_term_index);
                let query_terms = candidates
                    .iter()
                    .map(|(_query_term_index, candidate)| candidate.query_term.clone())
                    .collect();
                return Some(CandidateIntersection {
                    query_terms,
                    intersection,
                });
            }

            // Otherwise, increment the query term whose candidate had the smallest posting list.
            // That small posting list may have inhibited our ability to find an intersection
            // because it is small (just a heuristic).
            let iter = heap.pop();
            match iter {
                None => break,
                Some(mut iter) => {
                    iter.i += 1;
                    if iter.i >= iter.candidates.len() {
                        break;
                    }
                    heap.push(iter);
                }
            }
        }

        None
    }

    fn find_document_intersection(
        postings_list: Vec<&SkipList<Posting, ()>>,
    ) -> SkipList<Posting, ()> {
        let mut intersection = SkipList::new();
        for postings in postings_list.iter() {
            intersection = SkipList::intersect(&intersection, postings, true);
            if intersection.is_empty() {
                break;
            }
        }
        intersection
    }

    fn add_posting_to_hit(&self, posting: &Posting, hit: &mut Hit) {
        hit.fields
            .entry(posting.field_index)
            .or_insert_with(|| self.config.fields[posting.field_index].clone());

        let snippet = hit
            .field_snippets
            .entry(posting.field_index)
            .or_insert_with(|| Snippet {
                term_indices: Vec::new(),
                body: String::new(),
            });
        snippet.term_indices.push(posting.term_index);
    }

    fn compute_snippet_body(
        &self,
        document_id: u64,
        field_index: usize,
        hit_indices: &[usize],
    ) -> String {
        let pair = (document_id, field_index);
        let text = self
            .id_field_text
            .get(&pair)
            .expect("Each document field must have text");
        let terms = self
            .id_field_terms
            .get(&pair)
            .expect("Each document field must have terms");
        let mut ret = String::new();
        let mut h = 0;
        let window = 10;
        // To give context to user, show window of 10 words to left and 10 words to right of each
        // hit.
        let window_start = std::cmp::max(0, hit_indices[0] as isize - window as isize) as usize;
        let mut window_end = std::cmp::min(terms.len() - 1, hit_indices[0] + window);
        if window_start > 0 {
            ret.push_str("... ");
        }
        let mut prev_text_last = None;
        let mut i = window_start;
        while i < terms.len() {
            if i > window_end {
                if h >= hit_indices.len() {
                    if i < terms.len() - 1 {
                        ret.push_str(" ...");
                    }
                    break;
                }
                if i < hit_indices[h] - window {
                    // Skip ahead to beginning of window around next hit.
                    ret.push_str(" ... ");
                    prev_text_last = None;
                    i = hit_indices[h] - window;
                    window_end = hit_indices[h] + window;
                }
            }
            let is_hit = if h < hit_indices.len() && i == hit_indices[h] {
                // If we encounter a hit, expand window some more.
                window_end = i + window;
                h += 1;
                true
            } else {
                false
            };
            let term = &terms[i];
            // Add all text between last term and this one (spaces, punctuation, etc.)
            if let Some(prev_text_last) = prev_text_last {
                ret.push_str(&text[(prev_text_last + 1)..term.text_first]);
            }
            // Add emphasis tags around hit.
            if is_hit {
                ret.push_str("<em>");
            }
            // Add the term's original text
            ret.push_str(&text[term.text_first..=term.text_last]);
            if is_hit {
                ret.push_str("</em>")
            }
            prev_text_last = Some(term.text_last);
            i += 1;
        }
        ret
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_config() -> Config {
        Config { fields: vec![] }
    }

    #[test]
    fn indexes_documents_correctly() {
        let mut search_index = SearchIndex::new(Config {
            fields: vec![
                "name".to_string(),
                "locations".to_string(),
                "bio".to_string(),
            ],
        });
        let res: Result<serde_json::Value, _> = serde_json::from_str(
            r#"
            {
                "name": "Jane Smith",
                "locations": [
                    "Dayton, Ohio",
                    "Denver, Colorado"
                ],
                "bio": {
                    "title": "Jane Smith - Bio",
                    "text": "The great Jane was born and raised in the very neat Denver, Colorado"
                }
            }
        "#,
        );
        assert!(res.is_ok());
        let document_json = res.unwrap();

        search_index.index(123, &document_json);

        assert_eq!(16, search_index.term_postings.len());
        assert_eq!(1, search_index.id_terms.len());
        assert_eq!(1, search_index.id_postings.len());

        let jane_postings = search_index.term_postings.get(&"jane".to_string());
        assert!(jane_postings.is_some());
        let jane_postings = jane_postings.unwrap();
        assert_eq!(3, jane_postings.len());
        let mut iter = jane_postings.iter();
        assert_eq!(
            &Posting {
                document_id: 123,
                field_index: 0,
                term_index: 0
            },
            &iter.next().unwrap().key
        );
        // the terms from "text" come before those from "title" since "text" is alphabetically
        // smaller than "title".
        //
        // ie. terms are: "the great jane was born and raised in denver colorado jane smith's bio"
        assert_eq!(
            &Posting {
                document_id: 123,
                field_index: 2,
                term_index: 2
            },
            &iter.next().unwrap().key
        );
        assert_eq!(
            &Posting {
                document_id: 123,
                field_index: 2,
                term_index: 13
            },
            &iter.next().unwrap().key
        );
        assert!(iter.next().is_none());

        let terms = search_index.id_terms.get(&123);
        assert!(terms.is_some());
        let terms = terms.unwrap();
        assert_eq!(16, terms.len());

        let doc_postings = search_index.id_postings.get(&123);
        assert!(doc_postings.is_some());
        let doc_postings = doc_postings.unwrap();
        assert_eq!(22, doc_postings.len());

        let terms_for_trigram = search_index.trigram_terms.get(&"eat".to_string());
        assert!(terms_for_trigram.is_some());
        let terms_for_trigram = terms_for_trigram.unwrap();
        assert_eq!(2, terms_for_trigram.len());
        let mut terms_for_trigram: Vec<&String> = terms_for_trigram.iter().collect();
        terms_for_trigram.sort();
        assert_eq!("great", terms_for_trigram[0]);
        assert_eq!("neat", terms_for_trigram[1]);
    }

    #[test]
    fn performs_basic_insert_and_remove() {
        let mut search_index = SearchIndex::new(empty_config());

        // Insert posting
        let posting = Posting {
            field_index: 1,
            term_index: 2,
            document_id: 3,
        };
        search_index.insert_posting(&"foo".to_string(), posting);
        assert_eq!(1, search_index.term_postings.len());
        assert_eq!(1, search_index.id_terms.len());
        assert_eq!(1, search_index.id_postings.len());

        // Remove posting with document_id 3
        search_index.remove(3);
        assert_eq!(0, search_index.term_postings.len());
        assert_eq!(0, search_index.id_terms.len());
        assert_eq!(0, search_index.id_postings.len());
    }

    #[test]
    fn inserts_multiple_postings_for_same_term() {
        let mut search_index = SearchIndex::new(empty_config());

        let document_id = 123;

        let posting1 = Posting {
            field_index: 0,
            term_index: 0,
            document_id,
        };
        search_index.insert_posting(&"hello".to_string(), posting1);
        let posting2 = Posting {
            field_index: 0,
            term_index: 1,
            document_id,
        };
        search_index.insert_posting(&"hello".to_string(), posting2);

        assert_eq!(1, search_index.term_postings.len());
        assert_eq!(1, search_index.id_terms.len());
        assert_eq!(1, search_index.id_postings.len());

        assert_eq!(
            2,
            search_index
                .term_postings
                .get(&"hello".to_string())
                .unwrap()
                .len()
        );
        assert_eq!(1, search_index.id_terms.get(&document_id).unwrap().len());
        assert_eq!(2, search_index.id_postings.get(&document_id).unwrap().len());

        search_index.remove(document_id);
        assert_eq!(0, search_index.term_postings.len());
        assert_eq!(0, search_index.id_terms.len());
        assert_eq!(0, search_index.id_postings.len());
    }

    #[test]
    fn performs_simple_search() {
        let mut search_index = SearchIndex::new(Config {
            fields: vec!["name".to_string(), "bio".to_string()],
        });
        let doc1: Result<serde_json::Value, _> = serde_json::from_str(
            r#"
            {
                "name": "Jane Smith",
                "bio": "Jane is from the San Francisco Bay Area and serves as CEO of Foo Inc."
            }
        "#,
        );
        let doc2: Result<serde_json::Value, _> = serde_json::from_str(
            r#"
            {
                "name": "Samuel Wu",
                "bio": "The first memory Samuel has is of playing basketball with his mother in Manhattan"
            }
        "#,
        );
        search_index.index(1, &doc1.unwrap());
        search_index.index(2, &doc2.unwrap());

        let result = search_index.search("SAn francIsco");
        assert_eq!(result.spelling_correction, false);
        assert_eq!(result.query_terms, vec!["san", "francisco"]);
        let hits = result.hits;
        assert_eq!(hits.len(), 1);
        let hit = &hits[0];
        assert_eq!(hit.document_id, 1);
        assert_eq!(1, hit.fields.len());
        assert_eq!(hit.fields.get(&1).unwrap(), &"bio".to_string());
        assert_eq!(1, hit.field_snippets.len());
        let snippet = hit.field_snippets.get(&1).unwrap();
        assert_eq!(snippet.term_indices, vec![4, 5]);
        let expected_snippet_body = "Jane is from the <em>San</em> <em>Francisco</em> Bay Area and serves as CEO of Foo Inc";
        assert_eq!(snippet.body, expected_snippet_body);

        let result = search_index.search("basketball samuel");
        let hits = result.hits;
        assert_eq!(hits.len(), 1);
        let hit = &hits[0];
        assert_eq!(hit.document_id, 2);
        assert_eq!(2, hit.fields.len());
        assert_eq!(hit.fields.get(&0).unwrap(), &"name".to_string());
        assert_eq!(hit.fields.get(&1).unwrap(), &"bio".to_string());
        assert_eq!(2, hit.field_snippets.len());
        let snippet = hit.field_snippets.get(&0).unwrap();
        assert_eq!(snippet.term_indices, vec![0]);
        assert_eq!(snippet.body, "<em>Samuel</em> Wu".to_string());
        let snippet = hit.field_snippets.get(&1).unwrap();
        assert_eq!(snippet.term_indices, vec![3, 8]);
        assert_eq!(snippet.body, "The first memory <em>Samuel</em> has is of playing <em>basketball</em> with his mother in Manhattan".to_string());
    }

    #[test]
    fn generates_snippets_correctly_from_large_documents() {
        let mut search_index = SearchIndex::new(Config {
            fields: vec!["name".to_string(), "bio".to_string()],
        });
        let doc1: Result<serde_json::Value, _> = serde_json::from_str(
            r#"
            {
                "name": "Jane Smith",
                "bio": [
                    "Jane is from the San Francisco Bay Area and serves as CEO of Foo Inc.",
                    "She is on the board of Fortune 100 companies such as Facebook and Yahoo.",
                    "One of her passions is helping teams build skilled datacenter teams.",
                    "She graduated with a BS in Computer Science from Stanford University in 1998.",
                    "She found that she relished in the challenges of wrestling with difficult hardware problems.",
                    "Finally, she earned her PhD in Electrical Engineering in 2005.",
                    "One of her favorite pastimes is skiing at Lake Tahoe, where she spends almost each weekend."
                ]
            }
        "#,
        );
        let doc2: Result<serde_json::Value, _> = serde_json::from_str(
            r#"
            {
                "name": "Samuel Wu",
                "bio": "The first memory Samuel has is of playing basketball with his mother in Manhattan"
            }
        "#,
        );
        search_index.index(1, &doc1.unwrap());
        search_index.index(2, &doc2.unwrap());

        let result = search_index.search("CEO foo inc Computer science electrical engineering");
        assert_eq!(result.spelling_correction, false);
        assert_eq!(
            result.query_terms,
            vec![
                "ceo",
                "foo",
                "inc",
                "computer",
                "science",
                "electrical",
                "engineering"
            ]
        );
        let hits = result.hits;
        assert_eq!(hits.len(), 1);
        let hit = &hits[0];
        assert_eq!(hit.fields.len(), 1);
        assert_eq!(hit.fields.get(&1).unwrap(), &"bio".to_string());
        assert_eq!(hit.field_snippets.len(), 1);
        let snippet = hit.field_snippets.get(&1).unwrap();
        assert_eq!(snippet.term_indices, vec![11, 13, 14, 46, 47, 73, 74]);
        assert_eq!(
            snippet.body,
            "... is from the San Francisco Bay Area and serves as <em>CEO</em> of <em>Foo</em> \
             <em>Inc</em>. She is on the board of Fortune 100 companies such ... build skilled \
             datacenter teams. She graduated with a BS in <em>Computer</em> <em>Science</em> from \
             Stanford University in 1998. She found that she relished ... with difficult hardware \
             problems. Finally, she earned her PhD in <em>Electrical</em> <em>Engineering</em> in \
             2005. One of her favorite pastimes is skiing at ..."
                .to_string()
        );
    }

    #[test]
    fn performs_simple_spelling_correction() {
        let mut search_index = SearchIndex::new(Config {
            fields: vec!["name".to_string(), "bio".to_string()],
        });
        let doc1: Result<serde_json::Value, _> = serde_json::from_str(
            r#"
            {
                "name": "Jane Smith",
                "bio": [
                    "Jane is from the San Francisco Bay Area and serves as CEO of Foo Inc.",
                    "She is on the board of Fortune 100 companies such as Facebook and Yahoo.",
                    "One of her passions is helping teams build skilled datacenter teams.",
                    "She graduated with a BS in Computer Science from Stanford University in 1998.",
                    "She found that she relished in the challenges of wrestling with difficult hardware problems.",
                    "Finally, she earned her PhD in Electrical Engineering in 2005.",
                    "One of her favorite pastimes is skiing at Lake Tahoe, where she spends almost each weekend."
                ]
            }
        "#,
        );
        let doc2: Result<serde_json::Value, _> = serde_json::from_str(
            r#"
            {
                "name": "Samuel Wu",
                "bio": "The first memory Samuel has is of playing basketball with his mother in Manhattan"
            }
        "#,
        );
        search_index.index(1, &doc1.unwrap());
        search_index.index(2, &doc2.unwrap());

        let result = search_index.search("engineer electric");
        assert_eq!(result.spelling_correction, true);
        assert_eq!(result.query_terms, vec!["engineering", "electrical"]);
        let hits = result.hits;
        assert_eq!(hits.len(), 1);
        let hit = &hits[0];
        assert_eq!(hit.document_id, 1);
        assert_eq!(hit.fields.len(), 1);
        assert_eq!(hit.fields.get(&1).unwrap(), "bio");
        let snippet = hit.field_snippets.get(&1).unwrap();
        assert_eq!(snippet.term_indices, vec![73, 74]);
        assert_eq!(
            snippet.body,
            "... with difficult hardware problems. Finally, she earned her PhD in \
             <em>Electrical</em> <em>Engineering</em> in 2005. One of her favorite pastimes is \
             skiing at ..."
        );
    }
}
