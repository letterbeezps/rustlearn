use std::{cell::RefCell, collections::HashMap, rc::{Rc, Weak}};



#[derive(Debug)]
struct LruNode {
    key: i32,
    value: i32,
    pre: Weak<RefCell<LruNode>>,
    next:  Weak<RefCell<LruNode>>,
}

impl LruNode {
    fn new(key: i32, value: i32) -> Rc<RefCell<LruNode>> {
        Rc::new(RefCell::new(LruNode{
            key,
            value,
            pre: Weak::new(),
            next: Weak::new(),
        }))
    }
}

struct LRUCache {
    capacity: i32,
    dummy: Rc<RefCell<LruNode>>,
    map: HashMap<i32, Rc<RefCell<LruNode>>>,
}

impl LRUCache {

    fn new(capacity: i32) -> Self {
        let dummy = LruNode::new(0, 0);
        dummy.borrow_mut().pre = Rc::downgrade(&dummy);
        dummy.borrow_mut().next = Rc::downgrade(&dummy);
        Self {
            capacity,
            dummy,
            map: HashMap::new(),

        }
    }
    
    fn get(&mut self, key: i32) -> i32 {
        if let Some(node) = self.map.get(&key) {
            let node = Rc::clone(node);
            let value = node.borrow().value;
            self.remove(Rc::clone(&node));
            self.push_front(node);
            return value;
        }
        -1
    }
    
    fn put(&mut self, key: i32, value: i32) {
        if let Some(node) = self.map.get(&key) {
            let node = Rc::clone(node);
            node.borrow_mut().value = value;
            self.remove(Rc::clone(&node));
            self.push_front(node);
            return
        }
        let node = LruNode::new(key, value);
        self.map.insert(key, Rc::clone(&node));
        self.push_front(node);
        if self.map.len() > self.capacity as usize {
            let last_node = self.dummy.borrow().pre.upgrade().unwrap();
            self.map.remove(&last_node.borrow().key);
            self.remove(last_node);
        }
    }

    fn remove(&self, x: Rc<RefCell<LruNode>>) {
        let pre = x.borrow().pre.upgrade().unwrap();
        let next = x.borrow().next.upgrade().unwrap();
        pre.borrow_mut().next = Rc::downgrade(&next);
        next.borrow_mut().pre = Rc::downgrade(&pre);
    }

    fn push_front(&mut self, x: Rc<RefCell<LruNode>>) {
        let next = self.dummy.borrow().next.clone();
        next.upgrade().unwrap().borrow_mut().pre = Rc::downgrade(&x);
        x.borrow_mut().next = next;
        x.borrow_mut().pre = Rc::downgrade(&self.dummy);
        self.dummy.borrow_mut().next = Rc::downgrade(&x);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_lru() {
       let cache = LRUCache::new(3);
       println!("strong count after creating  = {}", Rc::strong_count(&cache.dummy));
       println!("weak count after creating  = {}", Rc::weak_count(&cache.dummy));
       println!("{:?}", &cache.dummy)
    }
}