#![allow(dead_code)]

extern crate askama;
#[macro_use]
extern crate serde_derive;

mod generational_arena;
mod search_index;
mod skip_list;

use std::collections::HashMap;

use actix_web::{http, middleware, server, App, Error, HttpResponse, Query, State};
use askama::Template;
use search_index::{Config, SearchIndex};

struct AppState {
    search_index: SearchIndex,
}

fn create_test_search_index() -> SearchIndex {
    let config = Config {
        fields: vec![
            "name".to_string(),
            "location".to_string(),
            "bio".to_string(),
        ],
    };
    let mut search_index = SearchIndex::new(config);

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

    search_index
}

#[derive(Template)]
#[template(path = "index.html")]
struct IndexTemplate {}

fn index((_state, _query): (State<AppState>, Query<HashMap<String, String>>)) -> Result<HttpResponse, Error> {
    let template = IndexTemplate {};
    Ok(HttpResponse::Ok()
        .content_type("text/html")
        .body(template.render().unwrap()))
}

fn search(
    (state, query): (State<AppState>, Query<HashMap<String, String>>),
) -> Result<HttpResponse, Error> {
    let response = if let Some(query) = query.get("query") {
        let result = state.search_index.search(query);
        HttpResponse::Ok().json(result)
    } else {
        HttpResponse::BadRequest()
            .content_type("application/json")
            .body("{\"error\":\"query required\"}")
    };
    Ok(response)
}

fn main() {
    ::std::env::set_var("RUST_LOG", "info,actix_web=info");
    env_logger::init();

    let sys = actix::System::new("search-index-testing");

    server::new(|| {
        let app_state = AppState {
            search_index: create_test_search_index(),
        };
        App::with_state(app_state)
            .middleware(middleware::Logger::default())
            .route("/", http::Method::GET, index)
            .route("/search", http::Method::GET, search)
    })
    .bind("127.0.0.1:8080")
    .unwrap()
    .start();
    let _ = sys.run();
}
