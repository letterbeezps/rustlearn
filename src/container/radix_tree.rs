
pub struct RadixTreeNode<T> {
    key: Vec<u8>,
    children: Vec<Box<RadixTreeNode<T>>>,
    value: Option<T>,
}

impl<T> Default for RadixTreeNode<T> {
    fn default() -> Self {
        Self {
            key: vec![],
            children: vec![],
            value: None,
        }
    }
}

impl<T> RadixTreeNode<T> {
    pub fn new() -> Self {
        Self {
            key: vec![],
            children: vec![],
            value: None,
        }
    }

    pub fn insert(&mut self, key: &[u8], value: T) {
        for child in self.children.iter_mut() {
            let prefix_len = common_prefix(&child.key, key);
            if prefix_len == 0 {
                continue;
            }
            if child.key.len() == prefix_len {
                if key.len() == prefix_len {
                    child.value = Some(value);
                } else {
                    child.insert(&key[prefix_len..], value);
                }
                return;
            } else { // split current child
                child.key = child.key[prefix_len..].to_owned();
                let mut parent_node = RadixTreeNode {
                    key: key[..prefix_len].to_owned(),
                    children: vec![std::mem::take(child)],
                    value: None,
                };
                // insert 
                if key.len() == prefix_len {
                    parent_node.value = Some(value);
                } else {
                    parent_node.insert(&key[prefix_len..], value);
                }
                *child = Box::new(parent_node);
                return;
            }
        }
        let new_node = RadixTreeNode {
            key: key.to_owned(),
            children: vec![],
            value: Some(value)
        };
        self.children.push(Box::new(new_node));
    }
}

fn common_prefix(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b.iter()).take_while(|&(a_char, b_char)| a_char == b_char).count()
}

#[test]
fn test_common_prefix() {
    let a = b"abcde";
    let b = b"abcee";
    let ret = common_prefix(a, b);
    assert_eq!(3, ret);
}