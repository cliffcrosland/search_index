#![allow(dead_code)]

extern crate serde_json;
extern crate unicode_segmentation;

mod analyzer;
mod generational_arena;
mod search_index;
mod skip_list;

use search_index::{Config, SearchIndex};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let config = Config {
        fields: vec![
            "name".to_string(),
            "location".to_string(),
            "bio".to_string(),
        ],
    };
    let index = SearchIndex::new(config);

    let query = args.join(" ");
    let hits = index.search(&query);
    dbg!(hits);
}
