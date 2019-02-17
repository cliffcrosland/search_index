// Represents an arena in which you can store items of a specific type.
//
// Items are stored in a vector of slots. When an item is inserted into the arena, it will be
// inserted (in constant time) into an empty slot, if one exists. If there are no empty slots, a
// new slot for the item will be added to the arena.
//
// When an item is inserted, an id is returned, which can be used to lookup the item later. The id
// consists of the index of the slot where the item was written, as well as the value of the
// generation counter of the slot. The generation counter allows us to immediately determine
// whether a 'get' request is attempting to read an item that has been deleted.
pub struct GenerationalArena<T> {
    // The arena in which to store items.
    arena: Vec<Slot<T>>,

    // Head of the free list.
    next_free: Option<usize>,

    // Number of items in the arena.
    len: usize,
}

// Represents an identifier allowing you to access an item stored in the arena.
//
// Returned from the insert operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GenerationalId {
    // Index of the item's slot in the arena vector.
    index: usize,

    // The value of the slot's generation counter when the item was inserted into the slot.
    generation: u16,
}

struct Slot<T> {
    value: Option<T>,
    generation: u16,
    next_free: Option<usize>,
}

impl<T> GenerationalArena<T> {
    pub fn new() -> GenerationalArena<T> {
        GenerationalArena {
            arena: Vec::new(),
            next_free: None,
            len: 0,
        }
    }

    // Insert a value into the arena. Return an id that can be used to look it up later.
    pub fn insert(&mut self, value: T) -> GenerationalId {
        self.len += 1;
        match self.next_free {
            // If there is a free slot, use it.
            Some(index) => {
                let slot = &mut self.arena[index];
                slot.generation += 1;
                slot.value = Some(value);
                self.next_free = slot.next_free;
                slot.next_free = None;
                GenerationalId {
                    index,
                    generation: slot.generation,
                }
            }

            // Otherwise, increase the size of the arena vector.
            None => {
                let index = self.arena.len();
                let slot = Slot {
                    value: Some(value),
                    generation: 0,
                    next_free: None,
                };
                self.arena.push(slot);
                GenerationalId {
                    index,
                    generation: 0,
                }
            }
        }
    }

    // Get a shared reference to the item matching the given id. If it does not exist, return None.
    pub fn get(&self, id: &GenerationalId) -> Option<&T> {
        if id.index >= self.arena.len() {
            return None;
        }
        let slot = &self.arena[id.index];
        if slot.generation != id.generation {
            return None;
        }
        slot.value.as_ref()
    }

    // Get an exclusive reference to the item matching the given id. If it does not exist, return
    // None.
    pub fn get_mut(&mut self, id: &GenerationalId) -> Option<&mut T> {
        if id.index >= self.arena.len() {
            return None;
        }
        let slot = &mut self.arena[id.index];
        if slot.generation != id.generation {
            return None;
        }
        slot.value.as_mut()
    }

    // Remove the item with the given id. Return true if item was found, false otherwise.
    pub fn remove(&mut self, id: &GenerationalId) -> bool {
        if id.index >= self.arena.len() {
            return false;
        }
        let slot = &mut self.arena[id.index];
        match slot.value {
            Some(_) => {
                self.len -= 1;
                slot.value = None;
                slot.next_free = self.next_free;
                self.next_free = Some(id.index);
                true
            }
            None => false,
        }
    }

    // Return the number of items stored.
    pub fn len(&self) -> usize {
        self.len
    }

    // Return the current number of slots in the arena.
    pub fn capacity(&self) -> usize {
        self.arena.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn performs_basic_operations_correctly() {
        let mut arena = GenerationalArena::new();

        // insert 3 items
        let id1 = arena.insert("Foo");
        assert_eq!(id1.index, 0);
        assert_eq!(id1.generation, 0);
        let id2 = arena.insert("Bar");
        assert_eq!(id2.index, 1);
        assert_eq!(id2.generation, 0);
        let id3 = arena.insert("Baz");
        assert_eq!(id3.index, 2);
        assert_eq!(id3.generation, 0);

        assert_eq!(arena.len(), 3);
        assert_eq!(arena.capacity(), 3);

        // read 3 items
        {
            let v1 = arena.get(&id1);
            assert!(v1.is_some());
            assert_eq!(v1.unwrap(), &"Foo");
            let v2 = arena.get(&id2);
            assert!(v2.is_some());
            assert_eq!(v2.unwrap(), &"Bar");
            let v3 = arena.get(&id3);
            assert!(v3.is_some());
            assert_eq!(v3.unwrap(), &"Baz");
        }

        // read out-of-range index
        let bad_id1 = GenerationalId {
            index: 123,
            generation: 0,
        };
        assert!(arena.get(&bad_id1).is_none());

        // read wrong generation
        let bad_id2 = GenerationalId {
            index: 0,
            generation: 2,
        };
        assert!(arena.get(&bad_id2).is_none());

        // Remove one of the items
        {
            let success = arena.remove(&id3);
            assert!(success);
            assert!(arena.get(&id3).is_none());
            assert_eq!(arena.len(), 2);
            assert_eq!(arena.capacity(), 3);
        }

        // Insert an item. Should replace the item that was just removed.
        {
            let id4 = arena.insert("Barz");
            assert_eq!(id4.index, 2);
            assert_eq!(id3.index, id4.index);
            assert_eq!(id4.generation, 1);
            assert_eq!(arena.len(), 3);
        }

        // Test exclusive reference get (i.e. get_mut)
        {
            let mut vec_arena = GenerationalArena::new();
            let id = vec_arena.insert(vec![1, 2, 3]);
            {
                let vec = vec_arena.get_mut(&id).unwrap();
                assert_eq!(vec.len(), 3);
                vec.push(4);
            }
            {
                let vec = vec_arena.get(&id).unwrap();
                assert_eq!(vec.len(), 4);
                for i in 0..=3 {
                    assert_eq!(vec[i], i + 1);
                }
            }
        }
    }
}
