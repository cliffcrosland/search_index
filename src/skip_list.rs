extern crate rand;

use self::rand::Rng;

use generational_arena::*;
use std::cmp::Ordering;
use std::fmt;

/// A skip list is a key-value store that allows you to visit entries in ascending order by key.
/// Insert, read, and delete operations execute in expected O(log n) time. Insert and remove use
/// expected O(log n) space.
///
/// The skip list is composed of multiple levels of linked lists that are sorted by key. Higher
/// levels have fewer nodes than lower levels. Search begins at the highest level, so it skips over
/// many nodes with each step, stepping down into lower levels as it hones in on the target.
///
/// When a new node is inserted, we first find where it should go in the bottom level, i.e. level
/// 0, in log(n) time. Then, we begin flipping a coin, promoting it into the next level every time
/// we flip "heads," stopping when we flip "tails." Hence, all entries appear in level 0, and, on
/// average, 1/2 of all nodes also appear in level 1, 1/4 in level 2, 1/8 in level 3, etc.
pub struct SkipList<K, V> where K: Key {
    head_id: NodeId,
    nodes: GenerationalArena<Node<K, V>>,
}

/// A key-value pair in the skip list.
pub struct SkipListEntry<K, V> where K: Key {
    pub key: K,
    pub value: V,
}

/// An iterator that allows you to scan through the elements in ascending order by key.
pub struct SkipListIterator<'a, K: 'a, V: 'a> where K: Key {
    cur: Option<NodeId>,
    nodes: &'a GenerationalArena<Node<K, V>>,
}

/// An iterator that allows you to scan through the elements whose keys match a specific prefix in
/// ascending order by key.
pub struct SkipListPrefixIterator<'a, K: 'a, V: 'a> where K: Key {
    prefix: K,
    cur: Option<NodeId>,
    nodes: &'a GenerationalArena<Node<K, V>>,
}

/// Represents the behavior that a key in the skip-list must have. Notably, the key must be
/// clonable and support both full comparison and prefix comparison.
pub trait Key: Clone + fmt::Debug {
    /// If prefix match disabled, return full comparison ordering.
    ///
    /// If prefix match enabled:
    /// - if query is a prefix of self, return Ordering::Equal.
    /// - otherwise, return full comparison ordering.
    fn key_cmp(&self, query: &Self, prefix_match: bool) -> Ordering;
}

/// Since it is common to use String keys, an implementation is provided here.
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

impl <'a, K> SearchParams<'a, K> where K: Key {
    pub fn new(target: &'a K) -> SearchParams<'a, K> {
        SearchParams {
            target,
            record_traversal: false,
            prefix_match: false,
        }
    }

    pub fn record_traversal(mut self) -> SearchParams<'a, K> {
        self.record_traversal = true;
        self
    }

    pub fn use_prefix_match(mut self) -> SearchParams<'a, K> {
        self.prefix_match = true;
        self
    }
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
    /// Constructs a new `SkipList`.
    pub fn new() -> SkipList<K, V> {
        let head = Node { entry: None, levels: vec![None] };
        let mut nodes = GenerationalArena::new();
        SkipList {
            head_id: nodes.insert(head),
            nodes,
        }
    }

    /// The number of entries in the skip list.
    pub fn len(&self) -> usize { 
        // head node not included in count
        self.nodes.len() - 1 
    }

    /// Set a key-value entry in the skip list. Return exclusive reference to the value. Useful for
    /// in-place modification after insert.
    pub fn set(&mut self, key: &K, value: V) {
        let params = SearchParams::new(key).record_traversal();
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
        let entry = SkipListEntry { key: (*key).clone(), value };
        let node_id = self.nodes.insert(Node {
            entry: Some(entry),
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

    /// Get a shared reference to the value that matches the key.
    pub fn get(&self, key: &K) -> Option<&V> {
        let params = SearchParams::new(key);
        let res = self.search(&params);
        if !res.success {
            return None;
        }
        let id = res.cur.unwrap();
        let node = self.nodes.get(&id).unwrap();
        let value = &node.entry.as_ref().unwrap().value;
        Some(value)
    }

    /// Get an exclusive reference to the value that matches the key. Useful for in-place
    /// modification.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let params = SearchParams::new(key);
        let res = self.search(&params);
        if !res.success {
            return None;
        }
        let id = res.cur.unwrap();
        let node = self.nodes.get_mut(&id).unwrap();
        let value = &mut node.entry.as_mut().unwrap().value; 
        Some(value)
    }

    /// Remove the entry corresponding to the given key. If the entry exists, remove it and return
    /// true. Otherwise, return false.
    pub fn remove(&mut self, key: &K) -> bool {
        let params = SearchParams::new(key).record_traversal();
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

    /// Return an iterator that visits all of the entries in the skip list in ascending order by
    /// key.
    pub fn iter(&self) -> SkipListIterator<K, V> {
        let head = self.nodes.get(&self.head_id).unwrap();
        self.iterator(head.levels[0])
    }

    /// Return an iterator that starts at the entry that matches the given key.
    pub fn iter_at(&self, key: &K) -> SkipListIterator<K, V> {
        let params = SearchParams::new(key);
        let res = self.search(&params);
        if res.success {
            self.iterator(res.cur)
        } else {
            self.iterator(None)
        }
    }

    /// Return an iterator that visits all of the entries that match the given prefix.
    pub fn prefix_iter_at(&self, prefix: &K) -> SkipListPrefixIterator<K, V> {
        let params = SearchParams::new(prefix).use_prefix_match();
        let res = self.search(&params);
        if res.success {
            self.prefix_iterator((*prefix).clone(), res.cur)
        } else {
            self.prefix_iterator((*prefix).clone(), None)
        }
    }

    /// Intersect two skip lists together, returning a new skip list.
    ///
    /// If `prefix_match` is false, return all entries from `a` and `b` whose keys are found in
    /// both lists.
    ///
    /// If `prefix_match` is true, return all entries from `a` and `b` as follows: if a key in `b`
    /// is a prefix of a key in `a`, we consider that an intersection. Let the key from `b` be
    /// called `key_b`. We then add each entry in `a` and `b` where `key_b` is a prefix of the
    /// entry's key.
    ///
    /// ## Example where `prefix_match == true` would be useful:
    ///
    /// Say we have user records that have `organization_id` and `user_id`, and say that we have
    /// multiple lists of user records. Say that we want to be able to quickly find the
    /// `organization_id` intersection of these lists of user records.
    ///
    /// To accomplish this, you could store these user records in skip lists and define the
    /// `key_cmp` function such that it only compares `organization_id` when `prefix_match` is
    /// true.
    ///
    /// TODO(cliff): `prefix_match` is awkward. Is there a better approach?
    ///
    pub fn intersect(a: &SkipList<K, V>, b: &SkipList<K, V>, prefix_match: bool) -> SkipList<K, V>
        where V: Clone {
        if a.len() > b.len() {
            return Self::intersect(b, a, prefix_match);
        }

        let mut ret = SkipList::new();
        if a.len() == 0 {
            for entry in b.iter() {
                ret.set(&entry.key, entry.value.clone());
            }
            return ret;
        }

        let mut a_cur_id = a.nodes.get(&a.head_id).unwrap().levels[0];
        let mut b_cur_id = b.nodes.get(&b.head_id).unwrap().levels[0];
        while a_cur_id.is_some() && b_cur_id.is_some() {
            let a_cur = a.nodes.get(&a_cur_id.unwrap()).unwrap();
            let b_cur = b.nodes.get(&b_cur_id.unwrap()).unwrap();
            let a_entry = a_cur.entry.as_ref().unwrap();
            let b_entry = b_cur.entry.as_ref().unwrap();
            match a_entry.key.key_cmp(&b_entry.key, prefix_match) {
                Ordering::Less => {
                    // Skip ahead as far as we can
                    let skip_ahead_level = Self::intersect_skip_ahead_level(a, a_cur,
                                                                            &b_entry.key,
                                                                            prefix_match);
                    a_cur_id = a_cur.levels[skip_ahead_level];
                },
                Ordering::Greater => {
                    // Skip ahead as far as we can
                    let skip_ahead_level = Self::intersect_skip_ahead_level(b, b_cur,
                                                                            &a_entry.key,
                                                                            prefix_match);
                    b_cur_id = b_cur.levels[skip_ahead_level];
                },
                Ordering::Equal => {
                    // Intersection found! Visit all entries in `a` and `b` that are equal to the
                    // current key under `key_cmp`. Add them all to the result.
                    let key = &b_entry.key;
                    while a_cur_id.is_some() {
                        let a_cur = a.nodes.get(&a_cur_id.unwrap()).unwrap();
                        let a_entry = a_cur.entry.as_ref().unwrap();
                        if a_entry.key.key_cmp(key, prefix_match) != Ordering::Equal {
                            break;
                        }
                        ret.set(&a_entry.key, a_entry.value.clone());
                        // advance by 1 entry (i.e. don't skip ahead using a higher level)
                        a_cur_id = a_cur.levels[0];
                    }
                    while b_cur_id.is_some() {
                        let b_cur = b.nodes.get(&b_cur_id.unwrap()).unwrap();
                        let b_entry = b_cur.entry.as_ref().unwrap();
                        if b_entry.key.key_cmp(key, prefix_match) != Ordering::Equal {
                            break;
                        }
                        ret.set(&b_entry.key, b_entry.value.clone());
                        // advance by 1 entry (i.e. don't skip ahead using a higher level)
                        b_cur_id = b_cur.levels[0];
                    }
                }
            }
        }

        ret
    }

    // Look at all of the skip pointers in `node.levels` and find the one that advances the
    // furthest without surpassing `advance_up_to_key`.
    fn intersect_skip_ahead_level(list: &SkipList<K, V>,
                                  node: &Node<K, V>,
                                  advance_up_to_key: &K,
                                  prefix_match: bool) -> usize {
        for level in 1..node.levels.len() {
            let next_node_id = node.levels[level];
            if next_node_id.is_none() {
                return level - 1;
            }
            let next_node_id = next_node_id.unwrap();
            let next_node = list.nodes.get(&next_node_id).unwrap();
            let next_entry = next_node.entry.as_ref().unwrap();
            match next_entry.key.key_cmp(advance_up_to_key, prefix_match) {
                Ordering::Equal => return level,
                Ordering::Greater => return level - 1,
                Ordering::Less => continue,
            }
        }
        0
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
            cur,
            nodes: &self.nodes,
        }
    }

    fn prefix_iterator(&self, prefix: K, cur: Option<NodeId>) -> SkipListPrefixIterator<K, V> {
        SkipListPrefixIterator {
            prefix,
            cur,
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
        self.cur?;
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
        self.cur?;
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
    fn performs_basic_operations_correctly() {
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

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct MergeTestKey {
        record_id: u64,
        field_id: u64,
        timestamp: u64,
    }

    impl MergeTestKey {
        fn new(record_id: u64, field_id: u64, timestamp: u64) -> MergeTestKey {
            MergeTestKey { record_id, field_id, timestamp }
        }
    }

    impl Key for MergeTestKey {
        fn key_cmp(&self, query: &MergeTestKey, prefix_match: bool) -> Ordering {
            if prefix_match {
                // Prefix match simply compares record_id
                return self.record_id.cmp(&query.record_id);
            }
            // Sorted by <record_id, field_id, timestamp>
            let mut ord = self.record_id.cmp(&query.record_id);
            if ord != Ordering::Equal {
                return ord;
            }
            ord = self.field_id.cmp(&query.field_id);
            if ord != Ordering::Equal {
                return ord;
            }
            self.timestamp.cmp(&query.timestamp)
        }
    }

    #[test]
    fn intersects_skip_lists_together_properly() {
        let mut a = SkipList::new();
        a.set(&MergeTestKey::new(1, 1, 1), ());
        a.set(&MergeTestKey::new(1, 2, 2), ());
        a.set(&MergeTestKey::new(1, 2, 3), ());
        a.set(&MergeTestKey::new(3, 1, 2), ());
        a.set(&MergeTestKey::new(3, 9, 2), ());
        a.set(&MergeTestKey::new(4, 1, 9), ());
        let mut b = SkipList::new();
        b.set(&MergeTestKey::new(1, 1, 5), ());
        b.set(&MergeTestKey::new(1, 2, 2), ());
        b.set(&MergeTestKey::new(2, 8, 3), ());
        b.set(&MergeTestKey::new(4, 1, 4), ());
        b.set(&MergeTestKey::new(5, 4, 4), ());

        // Intersect using full match. Only result should be key (1, 2, 2) since it is the only key
        // that appears in both lists.
        let full_match_merge_result = SkipList::intersect(&a, &b, false);
        assert_eq!(full_match_merge_result.len(), 1);
        let mut iter = full_match_merge_result.iter();
        let entry = iter.next();
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().key, MergeTestKey::new(1, 2, 2));
        assert!(iter.next().is_none());

        // Intersect using prefix match only. `key_cmp` is defined such that only `record_id` is
        // compared when `prefix_match` is set to true. Hence, the result should have all entries
        // that have `record_id` 1 or 4 since those are the only `record_ids` found in both lists.
        let prefix_merge_result = SkipList::intersect(&a, &b, true);
        assert_eq!(prefix_merge_result.len(), 6);
        let expected_keys = vec![
            MergeTestKey::new(1, 1, 1),
            MergeTestKey::new(1, 1, 5),
            MergeTestKey::new(1, 2, 2),
            MergeTestKey::new(1, 2, 3),
            MergeTestKey::new(4, 1, 4),
            MergeTestKey::new(4, 1, 9),
        ];
        let mut iter = prefix_merge_result.iter();
        let mut i = 0;
        while let Some(entry) = iter.next() {
            assert_eq!(entry.key, expected_keys[i]);
            i += 1;
        }
    }
}

#[cfg(all(feature = "benchmarks", test))]
mod benchmarks {
    use super::*;
    use self::test::Bencher;

    #[bench]
    fn bench_get(b: &mut Bencher) {
        let mut skip_list = SkipList::new();
        let value = "foo".to_string();
        for i in 0..1_000_000 {
            let key = i.to_string();
            skip_list.set(&key, value.clone());
        }
        let key = 123456.to_string();
        b.iter(|| skip_list.get(&key) );
    }

    #[bench]
    fn bench_set(b: &mut Bencher) {
        let n = 1_000_000;
        let mut pairs = Vec::new();
        let value = "foo".to_string();
        for i in 0..n {
            let key = i.to_string();
            pairs.push((key, value.clone()));
        }
        let mut skip_list = SkipList::new();
        let mut count = 0;
        b.iter(|| {
            let (key, value) = &pairs[count];
            skip_list.set(key, (*value).clone());
            count += 1;
            count %= pairs.len()
        });
    }

    #[bench]
    fn bench_iterate_1000(b: &mut Bencher) {
        let mut skip_list = SkipList::new();
        let n = 1000;
        let value = "foo".to_string();
        for i in 0..n {
            let key = i.to_string();
            skip_list.set(&key, value.clone());
        }
        b.iter(|| {
            let mut iter = skip_list.iter();
            while let Some(_entry) = iter.next() {}
        });
    }
}
