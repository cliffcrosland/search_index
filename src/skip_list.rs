extern crate rand;

use self::rand::Rng;

use generational_arena::*;
use std::cmp::Ordering;
use std::fmt;

// A skip list is a key-value store that allows you to visit entries in ascending order by key.
// Insert, read, and delete operations execute in expected O(log n) time. Insert and remove use
// expected O(log n) space.
//
// The skip list is composed of multiple levels of linked lists that are sorted by key. Higher
// levels have fewer nodes than lower levels. Search begins at the highest level, so it skips over
// many nodes with each step, stepping down into lower levels as it hones in on the target.
//
// When a new node is inserted, we first find where it should go in the bottom level, i.e. level 0,
// in log(n) time. Then, we begin flipping a coin, promoting it into the next level every time we
// flip "heads," stopping when we flip "tails." Hence, all entries appear in level 0, and, on
// average, 1/2 of all nodes also appear in level 1, 1/4 in level 2, 1/8 in level 3, etc.
pub struct SkipList<K, V> where K: Key {
    head_id: NodeId,
    nodes: GenerationalArena<Node<K, V>>,
}

// A key-value pair in the skip list.
pub struct SkipListEntry<K, V> where K: Key {
    key: K,
    value: V,
}

// An iterator that allows you to scan through the elements in ascending order by key.
pub struct SkipListIterator<'a, K: 'a, V: 'a> where K: Key {
    cur: Option<NodeId>,
    nodes: &'a GenerationalArena<Node<K, V>>,
}

// An iterator that allows you to scan through the elements whose keys match a specific prefix in
// ascending order by key.
pub struct SkipListPrefixIterator<'a, K: 'a, V: 'a> where K: Key {
    prefix: K,
    cur: Option<NodeId>,
    nodes: &'a GenerationalArena<Node<K, V>>,
}

// Represents the behavior that a key in the skip-list must have. Notably, the key must be clonable
// and support both full comparison and prefix comparison.
pub trait Key: Clone + fmt::Debug {
    // If prefix match disabled, return full comparison ordering.
    //
    // If prefix match enabled:
    // - if query is a prefix of self, return Ordering::Equal.
    // - otherwise, return full comparison ordering.
    fn key_cmp(&self, query: &Self, prefix_match: bool) -> Ordering;
}

// Since it is common to use String keys, an implementation is provided here.
impl Key for String {
    fn key_cmp(&self, query: &String, prefix_match: bool) -> Ordering {
        if !prefix_match {
            return self.cmp(query);
        }
        let mut self_iter = self.chars();
        let mut query_iter = query.chars();
        loop {
            let q = query_iter.next();
            // if query exhausted, query is a prefix of self
            if q.is_none() {
                return Ordering::Equal;
            }

            let s = self_iter.next();
            // if self exhausted, self is a prefix of query
            if s.is_none() {
                return Ordering::Less;
            }

            let sc = s.unwrap();
            let qc = q.unwrap();
            if sc < qc {
                return Ordering::Less;
            } else if sc > qc {
                return Ordering::Greater;
            }
        }
    }
}

// Represents a single node in the skip list. The head of the skip list is a special node that does
// not have a key-value entry, but all other nodes have a key-value entry.
struct Node<K, V> where K: Key {
    entry: Option<SkipListEntry<K, V>>,
    levels: Vec<Option<NodeId>>,
}

type NodeId = GenerationalId;

// Params to use when searching for a target. Can search for an exact match, or can search for the
// earliest key that has a given prefix.
struct SearchParams<'a, K: 'a> where K: Key {
    // The target to match.
    target: &'a K,

    // Whether we should record the last predecessor node found on each level during traversal.
    record_traversal: bool,

    // Whether to accept a prefix match on the target or an exact match.
    prefix_match: bool,
}

// The result of a search for a particular target. Returns the closest predecessor and the node
// that immediately follows the predecessor (which may match the target). Can optionally return the
// last predecessor node found on each level during traversal, which is helpful for writes.
struct SearchResult {
    // Whether the search found an entry that matched the target.
    success: bool,

    // The id of the predecessor node.
    prev_id: NodeId,

    // The id of the node that immediately follows the predecessor. If a node matched the target,
    // this will be its id.
    cur: Option<NodeId>,

    // A stack of the last predecessor node found on each level during traversal. Since we traverse
    // from the highest to lowest level, the vector elements here will be ordered from highest to
    // lowest level as well.
    traversal_stack: Vec<NodeId>,
}

impl <K, V> SkipList<K, V> where K: Key {
    pub fn new() -> SkipList<K, V> {
        let head = Node { entry: None, levels: vec![None] };
        let mut nodes = GenerationalArena::new();
        SkipList {
            head_id: nodes.insert(head),
            nodes: nodes,
        }
    }

    // The number of entries in the skip list.
    pub fn len(&self) -> usize { 
        // head node not included in count
        self.nodes.len() - 1 
    }

    // Set a key-value entry in the skip list.
    pub fn set(&mut self, key: &K, value: V) {
        let params = SearchParams { 
            target: key,
            record_traversal: true,
            prefix_match: false,
        };
        let res = self.search(&params);

        // If a matching entry was found, simply replace the value in the entry.
        if res.success {
            let id = res.cur.unwrap();
            let node = self.nodes.get_mut(&id).unwrap();
            let entry = node.entry.as_mut().unwrap();
            entry.value = value;
            return;
        } 

        // Otherwise, construct a new node.
        let node_id = self.nodes.insert(Node {
            entry: Some(SkipListEntry { key: (*key).clone(), value: value }),
            levels: Vec::new(),
        });

        // Insert node into the bottom level. Afterward, start flipping a coin. Promote the node
        // into the next level above every time you flip "heads." Stop when you flip "tails."
        let mut level = 0;
        loop {
            if level >= res.traversal_stack.len() {
                // If we are promoted into a level that does not exist yet, create new level.
                {
                    let head_node = self.nodes.get_mut(&self.head_id).unwrap();
                    head_node.levels.push(Some(node_id));
                }
                let node = self.nodes.get_mut(&node_id).unwrap();
                node.levels.push(None);
            } else {
                // Otherwise, insert the new node into the current level.
                let i = res.traversal_stack.len() - 1 - level;
                let prev_id = res.traversal_stack[i];
                let next_id: Option<NodeId>;
                {
                    let prev_node = self.nodes.get_mut(&prev_id).unwrap();
                    next_id = prev_node.levels[level];
                    prev_node.levels[level] = Some(node_id);
                }
                let node = self.nodes.get_mut(&node_id).unwrap();
                node.levels.push(next_id);
            }

            // Flip the coin.
            if rand::thread_rng().gen() {
                level += 1;
            } else {
                break;
            }
        }
    }

    // Get a shared reference to the value that matches the key.
    pub fn get(&self, key: &K) -> Option<&V> {
        let params = SearchParams { 
            target: key,
            record_traversal: false,
            prefix_match: false,
        };
        let res = self.search(&params);
        if !res.success {
            return None;
        }
        let id = res.cur.unwrap();
        let node = self.nodes.get(&id).unwrap();
        let value = &node.entry.as_ref().unwrap().value;
        Some(value)
    }

    // Get an exclusive reference to the value that matches the key.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let params = SearchParams { 
            target: key,
            record_traversal: false,
            prefix_match: false,
        };
        let res = self.search(&params);
        if !res.success {
            return None;
        }
        let id = res.cur.unwrap();
        let node = self.nodes.get_mut(&id).unwrap();
        let value = &mut node.entry.as_mut().unwrap().value; 
        Some(value)
    }

    // Remove the entry corresponding to the given key. If the entry exists, remove it and return
    // true. Otherwise, return false.
    pub fn remove(&mut self, key: &K) -> bool {
        let params = SearchParams { 
            target: key,
            record_traversal: true,
            prefix_match: false,
        };
        let res = self.search(&params);
        if !res.success {
            return false;
        }
        let node_id = res.cur.unwrap();
        let node_levels: Vec<Option<NodeId>>;
        {
            let node = self.nodes.get(&node_id).unwrap();
            node_levels = node.levels.clone();
        }
        for i in 0..res.traversal_stack.len() {
            let level = res.traversal_stack.len() - 1 - i;
            let prev_id = res.traversal_stack[i];
            let mut prev_node = self.nodes.get_mut(&prev_id).unwrap();
            {
                let next = prev_node.levels[level];
                if next.is_none() || next.unwrap() != node_id {
                    continue;
                }
            }
            prev_node.levels[level] = node_levels[level];
        }
        self.nodes.remove(&node_id);
        true
    }

    // Return an iterator that visits all of the entries in the skip list in ascending order by
    // key.
    pub fn iter(&self) -> SkipListIterator<K, V> {
        let head = self.nodes.get(&self.head_id).unwrap();
        self.iterator(head.levels[0])
    }

    // Return an iterator that starts at the entry that matches the given key.
    pub fn iter_at(&self, key: &K) -> SkipListIterator<K, V> {
        let params = SearchParams { 
            target: key,
            record_traversal: false,
            prefix_match: false,
        };
        let res = self.search(&params);
        match res.success {
            true => self.iterator(res.cur),
            false => self.iterator(None),
        }
    }

    // Return an iterator that visits all of the entries that match the given prefix.
    pub fn prefix_iter_at(&self, prefix: &K) -> SkipListPrefixIterator<K, V> {
        let params = SearchParams { 
            target: prefix,
            record_traversal: false,
            prefix_match: true,
        };
        let res = self.search(&params);
        match res.success {
            true => self.prefix_iterator((*prefix).clone(), res.cur),
            false => self.prefix_iterator((*prefix).clone(), None),
        }
    }

    #[allow(dead_code)]
    pub fn print(&self) {
        let mut cur = Some(self.head_id);
        while let Some(cur_id) = cur {
            let cur_node = self.nodes.get(&cur_id).unwrap();
            match &cur_node.entry {
                None => println!("Head, Levels: {:?}", cur_node.levels),
                Some(entry) => {
                    println!("Node id: {:?}, Key: {:?}, Levels: {:?}", 
                             cur_id, 
                             entry.key, 
                             cur_node.levels);
                },
            }
            cur = cur_node.levels[0];
        }
    }

    fn iterator(&self, cur: Option<NodeId>) -> SkipListIterator<K, V> {
        SkipListIterator {
            cur: cur,
            nodes: &self.nodes,
        }
    }

    fn prefix_iterator(&self, prefix: K, cur: Option<NodeId>) -> SkipListPrefixIterator<K, V> {
        SkipListPrefixIterator {
            prefix: prefix,
            cur: cur,
            nodes: &self.nodes,
        }
    }

    fn search(&self, params: &SearchParams<K>) -> SearchResult {
        let head_node = self.nodes.get(&self.head_id).unwrap();
        let mut level = (head_node.levels.len() - 1) as isize;
        let mut prev_id = self.head_id;
        let mut res = SearchResult {
            success: false,
            prev_id: self.head_id,
            traversal_stack: Vec::new(),
            cur: None,
        };

        // We must visit every level if: 
        // - we need to record the last node found on each level during traversal, or
        // - we need to find the earliest key that matches a prefix.
        let must_visit_every_level = params.record_traversal || params.prefix_match;

        while level >= 0 {
            res.prev_id = prev_id;
            let prev_node = self.nodes.get(&res.prev_id).unwrap();
            res.cur = prev_node.levels[level as usize];
            // Current key empty. Drop down a level and continue searching there.
            if res.cur.is_none() {
                if params.record_traversal { 
                    res.traversal_stack.push(prev_id); 
                }
                level -= 1;
                continue;
            }
            let cur_id = res.cur.unwrap();
            let cur_node = self.nodes.get(&cur_id).unwrap();
            let cur_key = &cur_node.entry.as_ref().unwrap().key;
            match cur_key.key_cmp(params.target, params.prefix_match) {
                // Current key less than target. Keep searching in same level.
                Ordering::Less => prev_id = cur_id,

                // Current key greater than target. Drop down a level and continue searching there.
                Ordering::Greater => {
                    if params.record_traversal { 
                        res.traversal_stack.push(prev_id); 
                    }
                    level -= 1;
                },

                // Current key matches target. Exit early if allowed. Otherwise, drop down a level
                // and continue searching there.
                Ordering::Equal => {
                    res.success = true;
                    if params.record_traversal {
                        res.traversal_stack.push(prev_id);
                    }
                    level -= 1;
                    if !must_visit_every_level {
                        break;
                    }
                },
            }
        }
        res
    }
}

impl <'a, K, V> Iterator for SkipListIterator<'a, K, V> where K: Key {
    type Item = &'a SkipListEntry<K, V>;

    fn next(&mut self) -> Option<&'a SkipListEntry<K, V>> {
        if self.cur.is_none() {
            return None;
        }
        let cur_id = self.cur.unwrap();
        let cur_node = self.nodes.get(&cur_id).unwrap();
        let res = cur_node.entry.as_ref();
        self.cur = cur_node.levels[0];
        res
    }
}

impl <'a, K, V> Iterator for SkipListPrefixIterator<'a, K, V> where K: Key {
    type Item = &'a SkipListEntry<K, V>;

    fn next(&mut self) -> Option<&'a SkipListEntry<K, V>> {
        if self.cur.is_none() {
            return None;
        }
        let cur_id = self.cur.unwrap();
        let cur_node = self.nodes.get(&cur_id).unwrap();
        let cur_key = &cur_node.entry.as_ref().unwrap().key;
        if cur_key.key_cmp(&self.prefix, true) != Ordering::Equal {
            return None;
        }
        let res = cur_node.entry.as_ref();
        self.cur = cur_node.levels[0];
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut skip_list = SkipList::new();
        let k1 = "apple".to_string();
        let k2 = "banana".to_string();
        let k3 = "bandana".to_string();
        let k4 = "cucumber".to_string();
        let k5 = "daisy".to_string();
        let v1 = "value1".to_string();
        let v2 = "value2".to_string();
        let v3 = "value3".to_string();
        let v4 = "value4".to_string();
        let v5 = "value5".to_string();

        // insert
        assert_eq!(skip_list.len(), 0);
        skip_list.set(&k1, v1.clone());
        skip_list.set(&k2, v2.clone());
        skip_list.set(&k3, v3.clone());
        skip_list.set(&k4, v4.clone());
        skip_list.set(&k5, v5.clone());
        assert_eq!(skip_list.len(), 5);

        // get values for some keys
        {
            let mut result = skip_list.get(&k1);
            assert!(result.is_some());
            assert_eq!(result.unwrap(), &v1);
            result = skip_list.get(&k3);
            assert!(result.is_some());
            assert_eq!(result.unwrap(), &v3);
        }

        // get unknown key should fail
        {
            let result = skip_list.get(&"unknown key".to_string());
            assert!(result.is_none());
        }

        // get mutable reference
        {
            let result = skip_list.get_mut(&k1);
            assert!(result.is_some());
            let value = result.unwrap();
            value.push_str("-updated");
        }

        // assert that value was changed
        {
            let result = skip_list.get(&k1);
            assert!(result.is_some());
            let value = result.unwrap();
            assert_eq!(value, &"value1-updated".to_string());
        }

        // undo change
        skip_list.set(&k1, v1.clone());

        // iterate through entire list in order
        {
            let mut iter = skip_list.iter();
            let mut entry = iter.next();
            assert!(entry.is_some());
            assert_eq!(entry.unwrap().key, k1);
            assert_eq!(entry.unwrap().value, v1);
            entry = iter.next();
            assert!(entry.is_some());
            assert_eq!(entry.unwrap().key, k2);
            assert_eq!(entry.unwrap().value, v2);
            entry = iter.next();
            assert!(entry.is_some());
            assert_eq!(entry.unwrap().key, k3);
            assert_eq!(entry.unwrap().value, v3);
            entry = iter.next();
            assert!(entry.is_some());
            assert_eq!(entry.unwrap().key, k4);
            assert_eq!(entry.unwrap().value, v4);
            entry = iter.next();
            assert!(entry.is_some());
            assert_eq!(entry.unwrap().key, k5);
            assert_eq!(entry.unwrap().value, v5);
            entry = iter.next();
            assert!(entry.is_none());
        }

        // iterate from specific key
        {
            let mut iter = skip_list.iter_at(&k2);
            let mut entry = iter.next();
            assert!(entry.is_some());
            assert_eq!(entry.unwrap().key, k2);
            assert_eq!(entry.unwrap().value, v2);
            entry = iter.next();
            assert!(entry.is_some());
            assert_eq!(entry.unwrap().key, k3);
            assert_eq!(entry.unwrap().value, v3);
            entry = iter.next();
            assert!(entry.is_some());
            assert_eq!(entry.unwrap().key, k4);
            assert_eq!(entry.unwrap().value, v4);
            entry = iter.next();
            assert!(entry.is_some());
            assert_eq!(entry.unwrap().key, k5);
            assert_eq!(entry.unwrap().value, v5);
            entry = iter.next();
            assert!(entry.is_none());
        }

        // iterating from unknown key should fail
        {
            let mut iter = skip_list.iter_at(&"unknown key".to_string());
            let entry = iter.next();
            assert!(entry.is_none());
        }

        // removing unknown key should fail
        {
            let success = skip_list.remove(&"unknown key".to_string());
            assert!(!success);
        }

        // remove key
        {
            let success = skip_list.remove(&k4);
            assert!(success);
        }
        assert_eq!(skip_list.len(), 4);

        // assert can no longer find the removed key
        {
            let result = skip_list.get(&k4);
            assert!(result.is_none());
            let mut iter = skip_list.iter_at(&k4);
            assert!(iter.next().is_none());
            let mut prefix_iter = skip_list.prefix_iter_at(&k4);
            assert!(prefix_iter.next().is_none());
        }

        // assert key no longer appears during full iteration
        {
            let mut iter = skip_list.iter();
            let mut entry = iter.next();
            assert!(entry.is_some());
            assert_eq!(entry.unwrap().key, k1);
            assert_eq!(entry.unwrap().value, v1);
            entry = iter.next();
            assert!(entry.is_some());
            assert_eq!(entry.unwrap().key, k2);
            assert_eq!(entry.unwrap().value, v2);
            entry = iter.next();
            assert!(entry.is_some());
            assert_eq!(entry.unwrap().key, k3);
            assert_eq!(entry.unwrap().value, v3);
            entry = iter.next();
            assert!(entry.is_some());
            assert_eq!(entry.unwrap().key, k5);
            assert_eq!(entry.unwrap().value, v5);
            entry = iter.next();
            assert!(entry.is_none());
        }

        // iterate through entries matching a prefix
        {
            let prefix = "ban".to_string();
            let mut iter = skip_list.prefix_iter_at(&prefix);
            // "banana"
            let mut entry = iter.next();
            assert!(entry.is_some());
            assert_eq!(entry.unwrap().key, k2);
            // "bandana"
            entry = iter.next();
            assert!(entry.is_some());
            assert_eq!(entry.unwrap().key, k3);
            // no more matches
            entry = iter.next();
            assert!(entry.is_none());
        }
    }
}
