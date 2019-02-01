use analyzer::Analyzer;
use skip_list::{SkipList, Key};

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use serde_json::{Value, Error};


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
struct SearchIndex {
    // Configuration for this search index.
    config: Config,

    // We use skip-lists below to compute intersections efficiently.

    // Term to Postings index
    term_postings: SkipList<String, SkipList<Posting, ()>>,

    // 3-Gram to Terms index
    trigram_terms: SkipList<String, SkipList<String, ()>>,

    // Soundex to Terms index
    soundex_terms: SkipList<String, SkipList<String, ()>>,

    // The following maps are useful for supporting the remove(document_id)
    // operation. Can quickly look up the terms and document postings to remove.

    // Map from document_id to terms found in the document.
    id_terms: HashMap<u64, HashSet<String>>,

    // Map from document_id to Posting structs.
    id_postings: HashMap<u64, HashSet<Posting>>,
}

struct Config {
    // Which fields to extract from the document JSON objects to be stored in
    // this index. The fields will be favored in the order they appear in this
    // vector. For example, if documents contain `bio`, `name`, and `location`
    // fields, and we want to favor `name` hits above `location` above `bio`,
    // then this vector should be `["name", "location", "bio"]`.
    fields: Vec<String>,
}

// Represents the location of a term in a document.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
            self.document_id.cmp(&query.document_id)
        } else {
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
}

// Represents a search result hit.
struct Hit {
    // The document where the search result was found
    document_id: u64,

    // String preview snippet of the match
    snippet: String,

    // Which terms from the query matched in the document
    query_term_matches: Vec<String>,

    // In which fields were the terms found
    field_indexes: Vec<usize>,
}

impl SearchIndex {
    pub fn new(config: Config) -> SearchIndex {
        SearchIndex {
            config: config,
            term_postings: SkipList::new(),
            trigram_terms: SkipList::new(),
            soundex_terms: SkipList::new(),
            id_terms: HashMap::new(),
            id_postings: HashMap::new(),
        }
    }

    pub fn index(&mut self, document_id: u64, document_json: &serde_json::Value) {
        for field_index in 0..self.config.fields.len() {
            let field = self.config.fields[field_index].clone();
            let terms = Analyzer::field_to_terms(document_json, &field);
            for term_index in 0..terms.len() {
                let term = &terms[term_index];
                let posting = Posting {
                    document_id: document_id,
                    field_index: field_index,
                    term_index: term_index,
                };
                self.insert(term, posting);
            }
        }
    }

    // TODO(cliff): Implement.
    pub fn search(&self, _query: &String) -> Vec<Hit> {
        Vec::new()
    }

    pub fn remove(&mut self, document_id: u64) {
        self.remove_term_postings(document_id);
        self.id_terms.remove(&document_id);
        self.id_postings.remove(&document_id);
    }

    fn insert(&mut self, term: &String, posting: Posting) {
        // 1. Update Term to Postings index
        let postings = self.term_postings.get_mut_with_default(term, || SkipList::new());
        postings.set(&posting, ());

        // 2. Update 3-Gram to Terms index
        for trigram in Analyzer::trigrams(term) {
            let mut terms = self.trigram_terms.get_mut_with_default(&trigram, || SkipList::new());
            terms.set(term, ());
        }

        // 3. Update Soundex to Terms index
        let soundex = Analyzer::soundex(term);
        let terms = self.soundex_terms.get_mut_with_default(&soundex, || SkipList::new());
        terms.set(term, ());

        let document_id = posting.document_id;

        // Insert into id_terms
        let terms = self.id_terms.entry(document_id).or_insert_with(|| HashSet::new());
        terms.insert((*term).clone());

        // Insert into id_postings
        let postings = self.id_postings.entry(document_id).or_insert_with(|| HashSet::new());
        postings.insert(posting);
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
                let mut term_postings = self.term_postings.get_mut(&term);
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
                self.term_postings.remove(&term);
            }
        }
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
                "bio".to_string()
            ],
        });
        let res: Result<Value, Error> = serde_json::from_str(r#"
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
        "#);
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
            &Posting { document_id: 123, field_index: 0, term_index: 0 }, 
            &iter.next().unwrap().key
        );
        // the terms from "text" come before those from "title" since "text" is alphabetically
        // smaller than "title".
        //
        // ie. terms are: "the great jane was born and raised in denver colorado jane smith's bio"
        assert_eq!(
            &Posting { document_id: 123, field_index: 2, term_index: 2 }, 
            &iter.next().unwrap().key
        );
        assert_eq!(
            &Posting { document_id: 123, field_index: 2, term_index: 13 }, 
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
        let mut iter = terms_for_trigram.iter();
        assert_eq!("great", iter.next().unwrap().key);
        assert_eq!("neat", iter.next().unwrap().key);
        assert!(iter.next().is_none());
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
            document_id: document_id,
        };
        search_index.insert(&"hello".to_string(), posting1);
        let posting2 = Posting {
            field_index: 0,
            term_index: 1,
            document_id: document_id,
        };
        search_index.insert(&"hello".to_string(), posting2);

        assert_eq!(1, search_index.term_postings.len());
        assert_eq!(1, search_index.id_terms.len());
        assert_eq!(1, search_index.id_postings.len());

        assert_eq!(2, search_index.term_postings.get(&"hello".to_string()).unwrap().len());
        assert_eq!(1, search_index.id_terms.get(&document_id).unwrap().len());
        assert_eq!(2, search_index.id_postings.get(&document_id).unwrap().len());

        search_index.remove(document_id);
        assert_eq!(0, search_index.term_postings.len());
        assert_eq!(0, search_index.id_terms.len());
        assert_eq!(0, search_index.id_postings.len());
    }
}
