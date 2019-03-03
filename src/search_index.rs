use analyzer::{Analyzer, Term};
use skip_list::{Key, SkipList};

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

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

// Represents a search result hit.
#[derive(Clone, Debug)]
pub struct Hit {
    // The document where the search result was found
    document_id: u64,

    // Pairs of (field_index, field_name) that matched query
    fields: HashMap<usize, String>,

    // Map from field_index (eg. "name", "bio", etc.) to snippets that show nearby text in source
    // document.
    field_snippets: HashMap<usize, Snippet>,
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
#[derive(Clone, Debug)]
pub struct Snippet {
    term_indices: Vec<usize>,
    body: String,
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
            self.id_field_terms
                .insert((document_id, field_index), terms.clone());
            for (term_index, term) in terms.iter().enumerate() {
                let posting = Posting {
                    document_id,
                    field_index,
                    term_index,
                };
                self.insert(&term.normalized, posting);
            }
        }
    }

    pub fn search(&self, query: &str) -> Vec<Hit> {
        // Analyze query. Translate to terms.
        let query_terms = Analyzer::text_to_terms(query);
        let query_terms = query_terms.iter().map(|t| &t.normalized);

        // Look up postings for each query term.
        let mut query_term_postings = Vec::with_capacity(query_terms.len());
        for (query_term_idx, query_term) in query_terms.enumerate() {
            if let Some(postings) = self.term_postings.get(query_term) {
                // If the query term appears in index, add its postings.
                query_term_postings.push((query_term_idx, postings));
            } else if let Some(correction) = self.recommend_spelling_correction(query_term) {
                // Recommend a spelling correction using trigrams and/or soundexes indices
                // ret.recommend_spelling_correction.push((query_term_idx, correction));
                dbg!(correction);
            }
        }

        // Find document_id intersection of all postings.
        query_term_postings.sort_by_key(|(_, postings)| postings.len());
        let mut intersection = SkipList::new();
        for (_query_term_idx, postings) in query_term_postings.into_iter() {
            intersection = SkipList::intersect(&intersection, postings, true);
        }

        //  Given postings, compute hits, summarizing search results for each document. Each hit
        //  shows field that matched and snippet showing surrounding text from original document.
        let mut hits: Vec<Hit> = Vec::new();

        for entry in intersection.iter() {
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
        for mut hit in hits.iter_mut() {
            for (field_index, ref mut snippet) in hit.field_snippets.iter_mut() {
                snippet.body =
                    self.compute_snippet_body(hit.document_id, *field_index, &snippet.term_indices);
            }
        }

        hits
    }

    pub fn remove(&mut self, document_id: u64) {
        self.remove_term_postings(document_id);
        self.id_terms.remove(&document_id);
        self.id_postings.remove(&document_id);
    }

    fn insert(&mut self, term: &str, posting: Posting) {
        // Update Term to Postings index
        let postings = self
            .term_postings
            .entry(term.to_string())
            .or_insert_with(SkipList::new);
        postings.set(&posting, ());

        // Update Trigram to Terms index
        for trigram in Analyzer::trigrams(term) {
            let mut terms = self
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
            let mut empty;
            {
                let mut term_postings = self.term_postings.get_mut(term);
                if term_postings.is_none() {
                    continue;
                }
                let mut term_postings = term_postings.unwrap();
                for posting in postings.iter() {
                    term_postings.remove(&posting);
                }
                empty = term_postings.len() == 0;
            }
            if empty {
                self.term_postings.remove(term);
            }
        }
    }

    fn recommend_spelling_correction(&self, term: &str) -> Option<String> {
        // First, look for good trigram match
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
                    let edit_distance = Analyzer::edit_distance(trigram_term, &term.to_string());
                    edit_distances.push((trigram_term, edit_distance));
                }
            }
        }
        // TODO(cliff): Return several spelling corrections, try running search with each, return the
        // most fruitful alternative query, etc.
        edit_distances.sort_by_key(|pair| pair.1);
        if edit_distances.is_empty() {
            None
        } else {
            Some(edit_distances[0].0.clone())
        }
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
        hit_indices: &Vec<usize>,
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
                    break;
                }
                if i < hit_indices[h] - window {
                    // Skip ahead to beginning of window around next hit.
                    ret.push_str(" ... ");
                    i = hit_indices[h] - window;
                    window_end = hit_indices[h] + window;
                }
            }
            let mut is_hit = false;
            if h < hit_indices.len() && i == hit_indices[h] {
                // If we encounter a hit, expand window some more.
                window_end = i + window;
                h += 1;
                is_hit = true;
            }
            let term = &terms[i];
            // Copy in all text between last term and this one (spaces, punctuation, etc.)
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
        search_index.insert(&"foo".to_string(), posting);
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
        search_index.insert(&"hello".to_string(), posting1);
        let posting2 = Posting {
            field_index: 0,
            term_index: 1,
            document_id,
        };
        search_index.insert(&"hello".to_string(), posting2);

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
                "bio": "Jane is from the San Francisco Bay Area and serves as CEO of Foo inc."
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

        let hits = search_index.search("san francisco");
        assert_eq!(hits.len(), 1);
        let hit = &hits[0];
        assert_eq!(hit.document_id, 1);
        assert_eq!(1, hit.fields.len());
        assert_eq!(hit.fields.get(&1).unwrap(), &"bio".to_string());
        assert_eq!(1, hit.field_snippets.len());
        let snippet = hit.field_snippets.get(&1).unwrap();
        assert_eq!(snippet.term_indices, vec![4, 5]);
        let expected_snippet_body = "Jane is from the <em>San</em> <em>Francisco</em> Bay Area and serves as CEO of Foo inc";
        assert_eq!(snippet.body, expected_snippet_body);
    }
}
