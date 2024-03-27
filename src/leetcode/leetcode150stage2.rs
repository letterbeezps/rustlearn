use std::{ cell::RefCell, rc::Rc};


#[derive(Debug, PartialEq, Eq)]
pub struct TreeNode {
  pub val: i32,
  pub left: Option<Rc<RefCell<TreeNode>>>,
  pub right: Option<Rc<RefCell<TreeNode>>>,
}

impl TreeNode {
  #[inline]
  pub fn new(val: i32) -> Self {
    TreeNode {
      val,
      left: None,
      right: None
    }
  }
}

pub fn max_depth(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    match root {
        None => 0,
        Some(ref node) => {
            max_depth(node.borrow().left.clone()).max(max_depth(node.borrow().right.clone()))
        }
    }
}

pub fn is_same_tree(p: Option<Rc<RefCell<TreeNode>>>, q: Option<Rc<RefCell<TreeNode>>>) -> bool {
    match (p, q) {
        (None, None) => true,
        (Some(ref p_node), Some(ref q_node)) => {
            if p_node.borrow().val != q_node.borrow().val {
                return false;
            }
            return 
            is_same_tree(p_node.borrow().left.clone(), q_node.borrow().left.clone()) 
            && 
            is_same_tree(p_node.borrow().right.clone(), q_node.borrow().right.clone());
        }
        _ => false
    }
}

pub fn invert_tree(root: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
    match root {
        None => None,
        Some(mut node) => {
            let left = invert_tree(node.borrow_mut().left.clone());
            let right = invert_tree(node.borrow_mut().right.clone());
            node.borrow_mut().left = right;
            node.borrow_mut().right = left;
            Some(node)
        }
    }
}

pub fn is_symmetric(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
    return is_symmetric_tree(
        root.as_ref().unwrap().borrow().left.clone(), 
        root.as_ref().unwrap().borrow().right.clone(),
    );
}

pub fn is_symmetric_tree(p: Option<Rc<RefCell<TreeNode>>>, q: Option<Rc<RefCell<TreeNode>>>) -> bool {
    match (p, q) {
        (None, None) => true,
        (Some(ref p_node), Some(ref q_node)) => {
            if p_node.borrow().val != q_node.borrow().val {
                return false;
            }
            return 
            is_symmetric_tree(p_node.borrow().left.clone(), q_node.borrow().right.clone()) 
            && 
            is_symmetric_tree(p_node.borrow().right.clone(), q_node.borrow().left.clone());
        }
        _ => false
    }
}

pub fn build_tree_preorder_inorder(preorder: Vec<i32>, inorder: Vec<i32>) -> Option<Rc<RefCell<TreeNode>>> {
    build_tree_preorder_inorder_dfs(0, preorder.len()-1, 0, inorder.len()-1, &preorder, &inorder)
}

fn build_tree_preorder_inorder_dfs(l1: usize, r1 :usize, l2: usize, r2: usize, preorder: &[i32], inorder: &[i32]) -> Option<Rc<RefCell<TreeNode>>> {
    if l1 <= r1 && l2 <= r2 {
        let mid = inorder.iter().position(|&x| x == preorder[l1]).unwrap();
        let lsize = mid - l2;
        let left = build_tree_preorder_inorder_dfs(l1+1, l1+lsize, l2, mid-1, preorder, inorder);
        let right = build_tree_preorder_inorder_dfs(l1+lsize+1, r1, mid+1, r2, preorder, inorder);
        return Some(Rc::new(RefCell::new(TreeNode{val: preorder[l1], left, right})));  
    }
    None
}

pub fn build_tree(inorder: Vec<i32>, postorder: Vec<i32>) -> Option<Rc<RefCell<TreeNode>>> {
    build_tree_postorder_inorder_dfs(0, postorder.len() as i32-1, 0, inorder.len() as i32-1, &postorder, &inorder)
}

fn build_tree_postorder_inorder_dfs(l1: i32, r1 :i32, l2: i32, r2: i32, postorder: &[i32], inorder: &[i32]) -> Option<Rc<RefCell<TreeNode>>> {
    if l1 <= r1 && l2 <= r2 {
        let mid = inorder.iter().position(|&x| x == postorder[r1 as usize]).unwrap() as i32;
        let lsize = mid - l2;
        let left = build_tree_postorder_inorder_dfs(l1, l1+lsize-1, l2, l2+lsize-1, postorder, inorder);
        let right = build_tree_postorder_inorder_dfs(l1+lsize, r1-1, mid+1, r2, postorder, inorder);
        return Some(Rc::new(RefCell::new(TreeNode{val: postorder[r1 as usize], left, right})));  
    }
    None
}

pub fn flatten(root: &mut Option<Rc<RefCell<TreeNode>>>) {
    if let Some(root) = root {
        let left = root.borrow_mut().left.take();
        if left.is_some() {
            let mut temp = left.clone();
            while temp.as_ref().unwrap().borrow().right.is_some() {
                let next = temp.as_mut().unwrap().borrow().right.clone();
                temp = next;
            }

            temp.unwrap().borrow_mut().right = root.borrow_mut().right.take();
            root.borrow_mut().right = left;
            root.borrow_mut().left = None;
        }

        flatten(&mut root.borrow_mut().right);
    }
}

pub fn has_path_sum(root: Option<Rc<RefCell<TreeNode>>>, target_sum: i32) -> bool {
    if root.is_none() {
        return false;
    }
    let val =  root.as_ref().unwrap().borrow().val;
    let left = root.as_ref().unwrap().borrow().left.clone();
    let right = root.as_ref().unwrap().borrow().right.clone();
    if left.is_none() && right.is_none() {
        return val == target_sum;
    }
    return 
    has_path_sum(left, target_sum - val)
    ||
    has_path_sum(right, target_sum - val)
    ;
}


pub fn sum_numbers(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    let mut ret = Vec::new();

    fn dfs(node: Option<Rc<RefCell<TreeNode>>>, cur: i32, ret: &mut Vec<i32>) {
        if node.is_none() {
            return;
        }
        let val = node.as_ref().unwrap().borrow().val;
        let cur = cur * 10 + val;
        let left = node.as_ref().unwrap().borrow().left.clone();
        let right = node.as_ref().unwrap().borrow().right.clone();
        if left.is_none() && right.is_none() {
            ret.push(cur);
            return;
        }
        dfs(left, cur, ret);
        dfs(right, cur, ret);
    }
    dfs(root, 0, &mut ret);
    ret.iter().sum()
}

pub fn max_path_sum(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    
    fn dfs(root: Option<Rc<RefCell<TreeNode>>>, res: &mut i32) -> i32 {
        if root.is_none() {
            return 0;
        }
        let val = root.as_ref().unwrap().borrow().val;
        let left = 0.max(dfs(root.as_ref().unwrap().borrow().left.clone(), res));
        let right = 0.max(dfs(root.as_ref().unwrap().borrow().right.clone(), res));

        *res = (*res).max(left+right+val);

        left.max(right) + val
    }

    let mut res = i32::MIN;
    dfs(root, &mut res);
    res
}


struct BSTIterator {
    stack: Vec<Option<Rc<RefCell<TreeNode>>>>,
}

impl BSTIterator {

    fn new(root: Option<Rc<RefCell<TreeNode>>>) -> Self {
        let mut stack = vec![];
        let mut root = root.clone();
        while root.is_some() {
            stack.push(root.clone());
            root = root.unwrap().borrow().left.clone();
        }
        Self{
            stack,
        }
    }
    
    fn next(&mut self) -> i32 {
        let tmp = self.stack.pop().unwrap();
        
        let mut p = tmp.as_ref().unwrap().borrow().right.clone();
        while p.is_some() {
            self.stack.push(p.clone());
            p = p.unwrap().borrow().left.clone();
        }
        tmp.unwrap().borrow().val
    }
    
    fn has_next(&self) -> bool {
        !self.stack.is_empty()
    }
}

pub fn count_nodes(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    if root.is_none() {
        return 0;
    }

    1 + count_nodes(root.as_ref().unwrap().borrow().left.clone()) + count_nodes(root.as_ref().unwrap().borrow().right.clone())
}

pub fn lowest_common_ancestor(root: Option<Rc<RefCell<TreeNode>>>, p: Option<Rc<RefCell<TreeNode>>>, q: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {

    match (root, p, q) {
        (Some(root), Some(p), Some(q)) => {
            if root.borrow().val == p.borrow().val || root.borrow().val == q.borrow().val {
                return Some(root);
            }
            let left = lowest_common_ancestor(root.borrow().left.clone(), Some(p.clone()), Some(q.clone()));
            let right = lowest_common_ancestor(root.borrow().right.clone(), Some(p.clone()), Some(q.clone()));

           if left.is_some() && right.is_some() {
            Some(root)
           } else {
            left.or(right)
           }
        }
        _ => None
    }
}

pub fn right_side_view(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
    let mut ret = vec![];
    fn dfs(root: Option<Rc<RefCell<TreeNode>>>,depth: i32, ret: &mut Vec<i32>) {
        if let Some(ref root) = root {
            if depth == ret.len() as i32 {
                ret.push(root.borrow().val);
            }
            
            dfs(root.borrow().right.clone(),depth+1, ret);
        
            dfs(root.borrow().left.clone(),depth+1, ret);
            
        }
    }
    dfs(root, 0, &mut ret);
    ret
}

pub fn average_of_levels(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<f64> {
    fn dfs(root: Option<Rc<RefCell<TreeNode>>>, depth: usize, count: &mut Vec<(i64, i64)>) {
        if let Some(ref root) = root {
            if depth >= count.len() {
                count.push((0, 0));
            }
            count[depth].0 += root.borrow().val as i64;
            count[depth].1 += 1;
            dfs(root.borrow().right.clone(),depth+1, count);
            dfs(root.borrow().left.clone(),depth+1, count);
        }
    }
    let mut count = vec![];
    dfs(root, 0, &mut count);
    count.into_iter().map(|(x, y)| x as f64 / y as f64).collect()
}

pub fn level_order(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Vec<i32>> {
    if root.is_none() {
        return vec![];
    }

    let mut ret = vec![];
    let mut level = vec![root.clone()];
    loop {
        let mut new_level = vec![];
        let mut val = vec![];
        for node in level {
            if let Some(ref node) = node {
                val.push(node.borrow().val);
                
                new_level.push(node.borrow().left.clone());
                new_level.push(node.borrow().right.clone());
                
            } 
        }
        if val.len() > 0 {
            ret.push(val);
        }
        if new_level.len() > 0 {
            level = new_level;
        } else {
            break;
        }
    }
    ret
}


pub fn zigzag_level_order(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Vec<i32>> {
    if root.is_none() {
        return vec![];
    }

    let mut ret = vec![];
    let mut level = vec![root.clone()];
    let mut depth = 0;
    loop {
        let mut new_level = vec![];
        let mut val = vec![];
        for node in level {
            if let Some(ref node) = node {
                val.push(node.borrow().val);
                new_level.push(node.borrow().left.clone());
                new_level.push(node.borrow().right.clone());
            } 
        }

        depth = (depth + 1) % 2;
        if val.len() > 0 {
            if depth == 0 {
                val.reverse();
            }
            ret.push(val);
        }
        if new_level.len() > 0 {
            level = new_level;
        } else {
            break;
        }
    }
    ret
}



