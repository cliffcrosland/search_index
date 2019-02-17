extern crate serde_json;
extern crate unicode_segmentation;

use std::{cmp, mem};

use serde_json::Value;
use unicode_segmentation::UnicodeSegmentation;

pub struct Analyzer;

/// Analyzes text and generates normalized strings that are useful to store in search index data
/// structures.
///
/// See `SearchIndex` for definitions of words like `Term`, `Trigram`, and `Soundex`.
///
/// # Supported operations:
///
/// - FieldToTerms(DocumentJson, Field) -> Vector of Terms
///
/// - `Trigrams(Term) -> Vector of Trigrams`
///
/// - `Soundex(Term) -> Soundex`
impl Analyzer {
    /// Read through the contents of `DocumentJson[Field]` and generate normalized string tokens
    /// known as Terms. If the field is a Json Object or Array, recursively traverse through the
    /// field and generate terms from each string you encounter along the way. If the field is a
    /// String, generate normalized tokens from it.
    pub fn field_to_terms(document_json: &Value, field: &str) -> Vec<String> {
        let mut ret = Vec::new();
        if let Value::Object(obj) = document_json {
            if let Some(v) = obj.get(field) {
                Self::field_to_terms_iter(v, &mut ret);
            }
        }
        ret
    }

    /// Read through the string `text` and generate normalized string tokens known as terms.
    pub fn text_to_terms(text: &str) -> Vec<String> {
        let mut ret = Vec::new();
        Self::text_to_terms_impl(text, &mut ret);
        ret
    }

    /// Get all of the substrings of length 3 that appear in the term. Additionally, include the
    /// start and end of a term.
    ///
    /// Example:
    /// ```
    /// "benevolent" =>
    /// ["$be", "ben", "ene", "nev", "evo", "vol", "ole", "len", "ent", "nt$"]
    /// ```
    pub fn trigrams(term: &str) -> Vec<String> {
        if term.is_empty() {
            return vec![];
        }
        let chars: Vec<char> = term.chars().collect();
        if term.len() == 1 {
            return vec![format!("${}$", chars[0])];
        }
        let mut ret = Vec::new();
        ret.push(format!("${}{}", chars[0], chars[1]));
        if chars.len() > 2 {
            for i in 0..(chars.len() - 2) {
                let mut trigram = String::with_capacity(3);
                for j in i..(i + 3) {
                    trigram.push(chars[j]);
                }
                ret.push(trigram);
            }
        }
        ret.push(format!(
            "{}{}$",
            chars[chars.len() - 2],
            chars[chars.len() - 1]
        ));
        ret
    }

    /// Return a four char string representing roughly how a term sounds. Particularly useful for
    /// matching person names. The algorithm can be found in 3.4 in "Chris Manning, Introduction to
    /// Information Retrieval, 1st Edition (Cambridge University Press, July 7, 2008)".
    ///
    /// Example:
    /// ```
    /// Soundex("herman") => "h655"
    /// ```
    /// The general idea is to assign letters that sound like one another to the same digit. The
    /// algorithm for computing English name soundexes is as
    /// follows:
    /// 1. Keep the first letter of the term.
    /// 2. Change these letters to the following digits:
    ///    - A, E, I, O, U, H, W, Y => 0
    ///    - B, F, P, V => 1
    ///    - C, G, J, K, Q, S, X, Z => 2
    ///    - D, T => 3
    ///    - L => 4
    ///    - M, N => 5
    ///    - R => 6
    /// 3. Coalesce repeated digits into one digit.
    /// 4. Remove all zeroes from the result. If less than 4 chars long, pad the end with zeroes.
    pub fn soundex(term: &str) -> String {
        if term.is_empty() {
            return String::new();
        }

        let chars: Vec<char> = term.chars().collect();
        let mut digits = Vec::with_capacity(chars.len());

        // Keep first letter
        digits.push(chars[0]);

        if chars.len() == 1 {
            return digits.into_iter().collect();
        }

        // Translate other letters into digits. Remove repeated digits.
        let mut prev_digit = None;
        for ch in chars.iter().skip(1) {
            let digit = match ch {
                'a' | 'e' | 'i' | 'o' | 'u' | 'h' | 'w' | 'y' => '0',
                'b' | 'f' | 'p' | 'v' => '1',
                'c' | 'g' | 'j' | 'k' | 'q' | 's' | 'x' | 'z' => '2',
                'd' | 't' => '3',
                'l' => '4',
                'm' | 'n' => '5',
                'r' => '6',
                _ => '0',
            };
            if prev_digit.is_some() && prev_digit.unwrap() == digit {
                continue;
            }
            digits.push(digit);
            prev_digit = Some(digit);
        }

        // Remove zeroes.
        let mut c = 1;
        for i in 1..digits.len() {
            if digits[i] != '0' {
                digits[c] = digits[i];
                c += 1;
            }
        }
        digits.truncate(c);

        // Resize to 4 chars, pad with zeroes on right
        digits.resize(4, '0');

        digits.into_iter().collect()
    }

    /// Compute Levenshtein Edit Distance between the given string values. Requires O(min(m, n))
    /// space and O(m * n) time.
    pub fn edit_distance(str1: &str, str2: &str) -> usize {
        let chars1: Vec<char> = str1.chars().collect();
        let chars2: Vec<char> = str2.chars().collect();
        if chars2.len() < chars1.len() {
            return Self::edit_distance(str2, str1);
        }
        let m = chars1.len();
        let n = chars2.len();
        let mut column1 = vec![0; m + 1];
        let mut column2 = vec![0; m + 1];
        for row in 0..=m {
            column1[row] = row;
        }
        for col in 1..=n {
            column2[0] = col;
            for row in 1..=m {
                let delete = column2[row - 1] + 1;
                let append = column1[row] + 1;
                let mut replace = column1[row - 1];
                if chars1[row - 1] != chars2[col - 1] {
                    replace += 1;
                }
                column2[row] = cmp::min(cmp::min(delete, append), replace)
            }
            mem::swap(&mut column1, &mut column2);
        }
        column1[m]
    }

    fn field_to_terms_iter(value: &Value, terms: &mut Vec<String>) {
        match value {
            Value::String(text) => Self::text_to_terms_impl(text, terms),
            Value::Array(arr) => {
                for val in arr {
                    Self::field_to_terms_iter(&val, terms);
                }
            }
            Value::Object(map) => {
                // Sort map entries by key. Add terms from each entry's value in order.
                let mut sorted_entries: Vec<(&String, &Value)> = map.iter().collect();
                sorted_entries.sort_by_key(|e| e.0);
                for (_, value) in sorted_entries {
                    Self::field_to_terms_iter(&value, terms);
                }
            }
            _ => {}
        }
    }

    fn text_to_terms_impl(text: &str, terms: &mut Vec<String>) {
        for word in text.unicode_words() {
            terms.push(word.to_lowercase());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use serde_json::Error;

    #[test]
    fn generates_terms_from_json_document_properly() {
        let res: Result<Value, Error> = serde_json::from_str(
            r#"
            {
                "name": "Jane Smith",
                "location": {
                    "city": "San Francisco",
                    "state": "CA",
                    "zip_code": "94123"
                },
                "bios": [
                    "Jane serves on the Board of Directors of the SF Symphony.",
                    "She is a composer and former professional violinist."
                ]
            }
        "#,
        );
        assert!(res.is_ok());
        let val = res.unwrap();

        // name
        assert_eq!(
            vec!["jane", "smith"],
            Analyzer::field_to_terms(&val, &"name".to_string())
        );

        // location. terms always gathered in ascending key order
        assert_eq!(
            vec!["san", "francisco", "ca", "94123"],
            Analyzer::field_to_terms(&val, &"location".to_string())
        );

        // bios
        let expected_bio_terms = vec![
            "jane",
            "serves",
            "on",
            "the",
            "board",
            "of",
            "directors",
            "of",
            "the",
            "sf",
            "symphony",
            "she",
            "is",
            "a",
            "composer",
            "and",
            "former",
            "professional",
            "violinist",
        ];
        assert_eq!(
            expected_bio_terms,
            Analyzer::field_to_terms(&val, &"bios".to_string())
        );
    }

    #[test]
    fn generates_terms_from_text_properly() {
        assert_eq!(
            vec!["the", "rain", "in", "spain", "stays", "mainly", "in", "the", "plain"],
            Analyzer::text_to_terms(&"The rain in spain stays mainly in the plain.".to_string())
        );

        let long_text = &r#"
            Fly me to the moon! Let me play among the stars.  
            Let me see what spring is like on Jupiter and Mars. 
            In other words, hold my hand.
        "#
        .to_string();
        let expected_terms = vec![
            "fly", "me", "to", "the", "moon", "let", "me", "play", "among", "the", "stars", "let",
            "me", "see", "what", "spring", "is", "like", "on", "jupiter", "and", "mars", "in",
            "other", "words", "hold", "my", "hand",
        ];
        assert_eq!(expected_terms, Analyzer::text_to_terms(long_text));
    }

    #[test]
    fn generates_trigrams_properly() {
        assert_eq!(Vec::<String>::new(), Analyzer::trigrams(&"".to_string()));
        assert_eq!(vec!["$a$"], Analyzer::trigrams(&"a".to_string()));
        assert_eq!(vec!["$ab", "ab$"], Analyzer::trigrams(&"ab".to_string()));
        assert_eq!(
            vec!["$cl", "cli", "lif", "iff", "ff$"],
            Analyzer::trigrams(&"cliff".to_string())
        );
    }

    #[test]
    fn generates_soundex_properly() {
        assert_eq!("h655", Analyzer::soundex(&"hermann".to_string()));
        assert_eq!("c410", Analyzer::soundex(&"cliff".to_string()));
        assert_eq!("s315", Analyzer::soundex(&"stephanie".to_string()));
    }

    #[test]
    fn computes_edit_distance() {
        assert_eq!(
            0,
            Analyzer::edit_distance(&"hello".to_string(), &"hello".to_string())
        );
        assert_eq!(
            3,
            Analyzer::edit_distance(&"he".to_string(), &"hello".to_string())
        );
        assert_eq!(
            4,
            Analyzer::edit_distance(&"hello".to_string(), &"world".to_string())
        );
        assert_eq!(
            3,
            Analyzer::edit_distance(&"carrot".to_string(), &"riot".to_string())
        );
        assert_eq!(
            3,
            Analyzer::edit_distance(&"foo".to_string(), &"bar".to_string())
        );
    }
}
