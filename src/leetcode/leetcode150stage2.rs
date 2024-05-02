use core::num;
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

pub fn get_minimum_difference(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {

    fn dfs(root: Option<Rc<RefCell<TreeNode>>>, ret: &mut i32, prev: &mut Option<i32>) {
        if let Some(ref root) = root {
            dfs(root.borrow().left.clone(), ret, prev);
            if let Some(prev) = *prev {
                *ret = (*ret).min(root.borrow().val - prev);
            }
            *prev = Some(root.borrow().val);
            dfs(root.borrow().right.clone(), ret, prev);
        }
    }

    let mut ret = i32::MAX;
    let mut prev: Option<i32> = None;
    dfs(root, &mut ret, &mut prev);
    ret
}


pub fn kth_smallest(root: Option<Rc<RefCell<TreeNode>>>, k: i32) -> i32 {
    fn dfs(root: Option<Rc<RefCell<TreeNode>>>, ret: &mut i32, rank: &mut i32, k: i32) {
        if let Some(ref root) = root {
            dfs(root.borrow().left.clone(), ret, rank, k);
            *rank += 1;
            if k == *rank {
                *ret = root.borrow().val;
                return;
            }
            dfs(root.borrow().right.clone(), ret, rank, k);
        }
    }

    let mut ret = 0;
    let mut rank = 0;
    dfs(root, &mut ret, &mut rank, k);
    ret
}

pub fn is_valid_bst(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
    fn dfs(root: Option<Rc<RefCell<TreeNode>>>, prev: &mut Option<i32>) -> bool {
        if let Some(ref root) = root {
            if !dfs(root.borrow().left.clone(), prev) {
                return false;
            }
            if let Some(prev) = *prev {
                if prev >= root.borrow().val {
                    return false;
                }
            }
            *prev = Some(root.borrow().val);
            dfs(root.borrow().right.clone(), prev)
        } else {
            true
        }
    }

    let mut prev: Option<i32> = None;
    dfs(root, &mut prev)
}

pub fn num_islands(grid: Vec<Vec<char>>) -> i32 {
        
    fn dfs(grid: &mut Vec<Vec<char>>, dx: &Vec<i32>, dy: &Vec<i32>, x: usize, y :usize, m: i32, n: i32) {
        grid[x][y] = '0';
        for i in 0..4 {
            let (a, b) = (dx[i] + x as i32, dy[i] + y as i32);
            if 0<=a && a<m && 0<=b && b<n && grid[a as usize][b as usize] == '1' {
                dfs(grid, dx, dy, a as usize, b as usize, m, n);
            }
        }
    }

    let (m, n) = (grid.len() as i32, grid[0].len() as i32);
    let dx = vec![0,-1,0,1];
    let dy = vec![1,0,-1,0];
    let mut res = 0;
    let mut grid = grid;
    for i in 0..m {
        for j in 0..n {
            if grid[i as usize][j as usize] == '1' {
                res += 1;
                dfs(&mut grid, &dx, &dy, i as usize, j as usize, m, n)
            }
        }
    }

    res
}


pub fn solve(grid: &mut Vec<Vec<char>>) {

    fn dfs(grid: &mut Vec<Vec<char>>, dx: &Vec<i32>, dy: &Vec<i32>, x: usize, y :usize, m: i32, n: i32) {
        if grid[x][y] != 'O'{
            return;
        }
        for i in 0..4 {
            let (a, b) = (dx[i] + x as i32, dy[i] + y as i32);
            if 0<=a && a<m && 0<=b && b<n && vec!['O', 'Y'].contains(&grid[a as usize][b as usize]) {
                grid[x][y] = 'Y';
                dfs(grid, dx, dy, a as usize, b as usize, m, n);
            }
        }
    }

    let (m, n) = (grid.len() as i32, grid[0].len() as i32);
    let dx = vec![0,-1,0,1];
    let dy = vec![1,0,-1,0];
    let mut res = 0;
    let mut grid = grid;
    for i in 0..m {
        for j in 0..n {
            if grid[i as usize][j as usize] == 'O' {
                dfs(&mut grid, &dx, &dy, i as usize, j as usize, m, n)
            }
        }
    }

    for i in 0..m {
        for j in 0..n {
            if grid[i as usize][j as usize] == 'Y' {
                grid[i as usize][j as usize] = 'X';
            }
        }
    }

}


pub fn solve_2(grid: &mut Vec<Vec<char>>) {
    fn dfs(grid: &mut Vec<Vec<char>>, dx: &Vec<i32>, dy: &Vec<i32>, x: usize, y :usize, m: i32, n: i32) {
        grid[x][y] = 'Y';
        for i in 0..4 {
            let (a, b) = (dx[i] + x as i32, dy[i] + y as i32);
            if 0<=a && a<m && 0<=b && b<n && grid[a as usize][b as usize] == 'O' {
                dfs(grid, dx, dy, a as usize, b as usize, m, n);
            }
        }
    }

    let (m, n) = (grid.len() as i32, grid[0].len() as i32);
    let dx = vec![0,-1,0,1];
    let dy = vec![1,0,-1,0];
    let mut grid = grid;
    for i in 0..m as usize {
        if grid[i][0] == 'O' {
            dfs(&mut grid, &dx, &dy, i, 0, m, n);
        }
        if grid[i][n as usize - 1] == 'O' {
            dfs(&mut grid, &dx, &dy, i, n as usize - 1, m, n);
        }
    }

    for j in 0..n as usize {
        if grid[0][j] == 'O' {
            dfs(&mut grid, &dx, &dy, 0, j, m, n);
        }
        if grid[m as usize - 1][j] == 'O' {
            dfs(&mut grid, &dx, &dy, m as usize - 1, j, m, n);
        }
    }

    for i in 0..m as usize {
        for j in 0..n as usize {
            if grid[i][j] == 'O' {
                grid[i][j] = 'X';
            }
            if grid[i][j] == 'Y' {
                grid[i][j] = 'O';
            }
        }
    }

}

pub fn calc_equation(equations: Vec<Vec<String>>, values: Vec<f64>, queries: Vec<Vec<String>>) -> Vec<f64> {
    use std::collections::{HashMap, HashSet};

    let mut graph: HashMap<String, Vec<(String, f64)>> = HashMap::new();
    let mut ans = vec![];

    for (i, v) in equations.into_iter().enumerate() {
        let v0 = graph.entry(v[0].clone()).or_insert(vec![]);
        v0.push((v[1].clone(), values[i]));
        let v1 = graph.entry(v[1].clone()).or_insert(vec![]);
        v1.push((v[0].clone(), 1. / values[i]));
    }

    fn dfs(graph: &HashMap<String, Vec<(String, f64)>>, visited: &mut HashSet<String>, start: &str, end: &str, ans: f64) -> f64 {
        if let Some(vec) = graph.get(start) {
            for (n, v) in vec {
                if visited.contains(n) {
                    continue;
                }
                visited.insert(n.clone());

                if n == end {
                    return  *v * ans;
                }
                let new_ans = dfs(graph, visited, n, end, ans * *v);
                if new_ans != -1. {
                    return new_ans;
                }
            }
        }
        -1.
    }

    for v in queries.iter() {
        let (start, end) = (&v[0], &v[1]);
        if !graph.contains_key(start) || !graph.contains_key(end) {
            ans.push(-1.);
            continue;
        }
        let mut visited = HashSet::new();
        ans.push(dfs(&graph, &mut visited, start, end, 1.))
    }

    ans
}


pub fn can_finish(num_courses: i32, prerequisites: Vec<Vec<i32>>) -> bool {
    use std::collections::HashMap;
    let mut graph = HashMap::new();
    for v in prerequisites.into_iter() {
        let v_child = graph.entry(v[1]).or_insert(vec![]);
        v_child.push(v[0]);
    }

    fn dfs(graph: &HashMap<i32, Vec<i32>>, visited: &mut HashMap<i32, i32>, cur: i32) -> bool {
        let v = visited.entry(cur).or_default();
        match *v {
            1 => return true,
            2 => return false,
            _ => {}
        }
        visited.insert(cur, 1);
        if let Some(child) = graph.get(&cur) {
            for c in child {
                if dfs(graph, visited, *c) {
                    return true;
                }
            }
        }
        visited.insert(cur, 2);
        false
    }

    let mut visited = HashMap::new();
    for i in (0..num_courses).into_iter().rev() {
        if dfs(&graph, &mut visited, i) {
            return false;
        }
    }

    true
}


pub fn find_order(num_courses: i32, prerequisites: Vec<Vec<i32>>) -> Vec<i32> {
    use std::collections::HashMap;

    let mut graph = HashMap::new();
    for v in prerequisites.into_iter() {
        let v_child = graph.entry(v[1]).or_insert(vec![]);
        v_child.push(v[0]);
    }

    fn dfs(graph: &HashMap<i32, Vec<i32>>, visited: &mut HashMap<i32, i32>, cur: i32, ans: &mut Vec<i32>) -> bool {
        let v = visited.entry(cur).or_default();
        match *v {
            1 => return true,
            2 => return false,
            _ => {}
        }
        visited.insert(cur, 1);
        if let Some(child) = graph.get(&cur) {
            for c in child {
                if dfs(graph, visited, *c, ans) {
                    return true;
                }
            }
        }
        visited.insert(cur, 2);
        ans.push(cur);
        false
    }

    let mut ans = vec![];
    let mut visited = HashMap::new();
    for i in 0..num_courses {
        if dfs(&graph, &mut visited, i, &mut ans) {
            return vec![];
        }
    }

    ans.reverse();
    ans
}

pub fn snakes_and_ladders(board: Vec<Vec<i32>>) -> i32 {
    use std::collections::{VecDeque, HashSet};
    let graph: Vec<i32> = board
        .into_iter()
        .rev()
        .enumerate()
        .map(|(i, mut v)| if i % 2 == 0 { v } else { v.reverse(); v })
        .flatten()
        .collect();
    let length = graph.len();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back((0, 0));


    while let Some((i, step)) = queue.pop_front() {
        if i == length - 1 {
            return  step;
        }

        for j in i+1..=(i+6).min(length-1) {
            let next = if graph[j] == -1 {
                j
            } else {
                graph[j] as usize - 1
            };

            if !visited.contains(&next) {
                queue.push_back((next, step+1));
                visited.insert(next);
            }
        }
    }

    -1
}

pub fn min_mutation(start_gene: String, end_gene: String, bank: Vec<String>) -> i32 {
    use std::collections::{VecDeque, HashSet};
    fn is_reached(s1: &String, s2: &String) -> bool {
        let cnt = s1.chars().zip(s2.chars()).fold(
            0, |cnt, (ch1, ch2)| 
            if ch1 == ch2 { cnt } else { cnt + 1 });
        cnt == 1
    }

    let mut queue = VecDeque::new();
    let mut visited = HashSet::new();
    queue.push_back((&start_gene, 0));
    visited.insert(&start_gene);
    while let Some((s, step)) = queue.pop_front() {
        if s == &end_gene {
            return step;
        }
        for next_s in bank.iter() {
            if !visited.contains(next_s) && is_reached(s, next_s) {
                queue.push_back((next_s, step+1));
                visited.insert(next_s);
            }
        }
    }
    -1
}

pub fn ladder_length(begin_word: String, end_word: String, word_list: Vec<String>) -> i32 {
    use std::collections::{VecDeque, HashSet};
    fn is_reached(s1: &String, s2: &String) -> bool {
        let cnt = s1.chars().zip(s2.chars()).fold(
            0, |cnt, (ch1, ch2)| 
            if ch1 == ch2 { cnt } else { cnt + 1 });
        cnt == 1
    }

    let mut queue = VecDeque::new();
    let mut visited = HashSet::new();
    queue.push_back((&begin_word, 0));
    visited.insert(&begin_word);
    while let Some((s, step)) = queue.pop_front() {
        if s == &end_word {
            return step+1;
        }
        for next_s in word_list.iter() {
            if !visited.contains(next_s) && is_reached(s, next_s) {
                queue.push_back((next_s, step+1));
                visited.insert(next_s);
            }
        }
    }
    0
}

pub fn letter_combinations(digits: String) -> Vec<String> {
    let m = vec!["","","abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"];

    let mut ret = vec![];
    fn dfs(m: &Vec<&str>, digits: &str, ret: &mut Vec<String>, d: usize, cur: String) {
        if d == digits.len() {
            ret.push(cur.to_string());
            return;
        }
        let cur_index = (digits.as_bytes()[d] - b'0') as usize;
        for c in m[cur_index].chars() {
            dfs(m, digits, ret, d+1, cur.clone() + &c.to_string())
        }
    }
    if digits.is_empty() {
        return vec![];
    }
    dfs(&m, &digits, &mut ret, 0, "".to_string());
    ret
}

pub fn combine(n: i32, k: i32) -> Vec<Vec<i32>> {
    let mut path = vec![];
    let mut ret = vec![];

    fn dfs(n: i32, k: i32, start: i32, path: &mut Vec<i32>, ret: &mut Vec<Vec<i32>>) {
        if path.len() == k as usize {
            ret.push(path.clone());
            return;
        }
        for i in start..=(n-(k-path.len() as i32))+1 {
            path.push(i);
            dfs(n, k, i+1, path, ret);
            path.pop();
        }
    }
    dfs(n, k, 1, &mut path, &mut ret);
    ret
}

pub fn permute(nums: Vec<i32>) -> Vec<Vec<i32>> {
    use std::collections::HashSet;
    fn dfs(nums: &Vec<i32>, ret: &mut Vec<Vec<i32>>, path: &mut Vec<i32>, visited: &mut HashSet<i32>, length: usize) {
        if length == nums.len() {
            ret.push(path.clone());
            return;
        }
        for n in nums {
            if !visited.contains(n) {
                visited.insert(*n);
                path.push(*n);

                dfs(nums, ret, path, visited, length+1);

                path.pop();
                visited.remove(n);
            }
        }
    }

    let mut ret = vec![];
    let mut path = vec![];
    let mut visited = HashSet::new();
    dfs(&nums, &mut ret, &mut path, &mut visited, 0);
    ret
}

pub fn combination_sum(candidates: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
    let mut candidates = candidates;
    candidates.sort();

    fn dfs(candidates: &Vec<i32>, target: i32, index: usize, path: &mut Vec<i32>, ret: &mut Vec<Vec<i32>>) {
        if target == 0 {
            ret.push(path.clone());
            return;
        }
        if target > 0 {
            for i in index..candidates.len() {
                if candidates[i] > target {
                    break;
                }
                path.push(candidates[i]);
                dfs(candidates, target - candidates[i], i, path, ret);
                path.pop();
            }
        }
    }

    let mut ret = vec![];
    let mut path = vec![];
    dfs(&candidates, target, 0, &mut path, &mut ret);
    ret
}

pub fn total_n_queens(n: i32) -> i32 {

    fn dfs(u: usize, n: usize, res: &mut i32, col: &mut Vec<bool>, diag: &mut Vec<bool>, ant_diag: &mut Vec<bool>) {
        if u == n {
            *res += 1;
            return;
        }
        for i in 0..n {
            if !col[i] && !diag[u+n-i] && !ant_diag[u+i] {
                col[i] = true;
                diag[u+n-i] = true;
                ant_diag[u+i] = true;

                dfs(u+1, n, res, col, diag, ant_diag);

                col[i] = false;
                diag[u+n-i] = false;
                ant_diag[u+i] = false;
            }
        }
    }

    let n = n as usize;
    let mut res = 0;
    let mut col = vec![false; n];
    let mut diag = vec![false; 2*n];
    let mut ant_diag = vec![false; 2*n];

    dfs(0, n, &mut res, &mut col, &mut diag, &mut ant_diag);
    res
}

pub fn generate_parenthesis(n: i32) -> Vec<String> {

    fn dfs(cur: &str, left: usize, right: usize, n: usize, res: &mut Vec<String>) {
        if right == n {
            res.push(cur.to_string());
            return;
        }
        if left < n {
            dfs(&(cur.to_string() + "("), left+1, right, n, res);
        }
        if right < left {
            dfs(&(cur.to_string() + ")"), left, right+1, n, res);
        }
    }
    let mut res = vec![];
    dfs("", 0, 0, n as usize, &mut res);
    res
}

pub fn exist(board: Vec<Vec<char>>, word: String) -> bool {

    fn dfs(x: usize, y: usize, u: usize, row: usize, col: usize, 
        board: &Vec<Vec<char>>,
        dx: &Vec<i32>,
        dy: &Vec<i32>,
        visited: &mut Vec<Vec<bool>>,
        word: &Vec<char>
    ) -> bool {
        if u == word.len() {
            return true;
        }
        visited[x][y] = true;
        for i in 0..4 as usize {
            let (a, b) = (x as i32 +dx[i], y as i32 +dy[i]);
            if a>=0 && a<row as i32 && b>=0 && b<col as i32 && !visited[a as usize][b as usize] && board[a as usize][b as usize] == word[u] {
                if dfs(a as usize, b as usize, u+1, row, col, board, dx, dy, visited, word) {
                    return true;
                }
            }
        }
        
        visited[x][y] = false;
        false

    }
    let word = word.chars().collect::<Vec<char>>();
    let (row, col) = (board.len(), board[0].len());
    let mut visited = vec![vec![false; col]; row];
    let dx = vec![-1, 0, 1, 0];
    let dy = vec![0, 1, 0, -1];
    for i in 0..row {
        for j in 0..col {
            if board[i][j] == word[0] {
                if dfs(i, j, 1, row, col, &board, &dx, &dy, &mut visited, &word) {
                    return true;
                }
            }
        }
    }
    false
}

use std::collections::HashMap;
#[derive(Clone)]
struct Trie {
    node: Rc<RefCell<TrieNode>>
}

struct TrieNode {
    val: Option<bool>,
    children: HashMap<char,Rc<RefCell<TrieNode>>>
}


impl Trie {

    fn new() -> Self {
        Self{
            node: Rc::new(RefCell::new(TrieNode {
                val:None,
                children: HashMap::new(),
            }))
        }
    }
    
    fn insert(&self, word: String) {
        let mut cur = self.node.clone(); 
        for c in word.chars() {
            if let Some(node) = cur.clone().borrow().children.get(&c){
                cur = node.clone();
                continue;
            }
            {
                let new_node = Rc::new(RefCell::new(TrieNode{
                    val: None,
                    children: HashMap::new(),
                }));
                cur.borrow_mut().children.insert(c, new_node.clone());
                cur = new_node.clone();
            }
        }
        cur.borrow_mut().val = Some(true);
    }
    
    fn search(&self, word: String) -> bool {
    
        let mut cur = self.node.clone(); 
        for c in word.chars() {
            if let Some(node) = cur.clone().borrow().children.get(&c) {
                cur = node.clone();
                continue;
            }
            return false;
        }
        cur.clone().borrow().val.is_some()
    }
    
    fn starts_with(&self, prefix: String) -> bool {
        let mut cur = self.node.clone(); 
        for c in prefix.chars() {
            if let Some(node) = cur.clone().borrow().children.get(&c) {
                cur = node.clone();
                continue;
            }
            return false;
        }
        true
    }
}


struct WordDictionary {
    node: Rc<RefCell<TrieNode>>
}

impl WordDictionary {
    
    fn new() -> Self {
        Self{
            node: Rc::new(RefCell::new(TrieNode {
                val:None,
                children: HashMap::new(),
            }))
        }
    }
    
    fn add_word(&self, word: String) {
        let mut cur = self.node.clone(); 
        for c in word.chars() {
            if let Some(node) = cur.clone().borrow().children.get(&c){
                cur = node.clone();
                continue;
            }
            {
                let new_node = Rc::new(RefCell::new(TrieNode{
                    val: None,
                    children: HashMap::new(),
                }));
                cur.borrow_mut().children.insert(c, new_node.clone());
                cur = new_node.clone();
            }
        }
        cur.borrow_mut().val = Some(true);
    }
    
    fn search(&self, word: String) -> bool {
        let word = word.chars().collect::<Vec<char>>();
        Self::search_help(self.node.clone(), &word, 0)
    }

    fn search_help(node: Rc<RefCell<TrieNode>>, word: &Vec<char>, index: usize) -> bool {
        if index == word.len() {
            return node.clone().borrow().val.is_some();
        }
        let c = word[index];
        if c != '.' {
            if let Some(child) = node.clone().borrow().children.get(&c) {
                return Self::search_help(child.clone(), word, index+1)
            }
        } else {
            if node.borrow().children
            .iter()
            .any(|(_, child)| {
                Self::search_help(child.clone(), word, index+1)
            }) {
                return true;
            }
        }
        false
    }
}


pub fn find_words(board: Vec<Vec<char>>, words: Vec<String>) -> Vec<String> {
    let trie = Trie::new();
    for word in words {
        trie.insert(word);
    }
    let mut ret = vec![];
    let (m, n) = (board.len(), board[0].len());
    let dx = vec![-1, 0, 1, 0];
    let dy = vec![0, 1, 0, -1];
    
    fn dfs(
        board: &mut Vec<Vec<char>>, 
        node: Rc<RefCell<TrieNode>>,
        path: &str,
        dx: &Vec<i32>,
        dy: &Vec<i32>,
        ret: &mut Vec<String>,
        i: usize,
        j: usize,
        m: usize,
        n: usize,

    ) {
        if node.clone().borrow().val.is_some() {
            ret.push(path.to_string());
            node.borrow_mut().val = None;
        }
        let tmp = board[i][j];
        println!("tmp: {:?}", tmp);
        if let Some(child) = node.clone().borrow().children.get(&tmp) {
            if child.clone().borrow().val.is_some() {
                ret.push(path.to_string() + &tmp.to_string());
                child.borrow_mut().val = None;
            }
            board[i][j] = '#';
            for index in 0..4 {
                let (a, b) = (i as i32 +dx[index], j as i32 +dy[index]);
                if a>=0 && a<m as i32 && b>=0 && b<n as i32 {
                    dfs(board, child.clone(), &(path.to_string() + &tmp.to_string()), dx, dy, ret, a as usize, b as usize, m, n)
                }
            }
            board[i][j] = tmp;
        }
    }

    let mut board = board;
    for i in 0..m {
        for j in 0..n {
            dfs(&mut board, trie.node.clone(), "", &dx, &dy, &mut ret, i, j, m, n);
        }
    }

    ret
}

pub fn sorted_array_to_bst(nums: Vec<i32>) -> Option<Rc<RefCell<TreeNode>>> {
    fn dfs(nums: &Vec<i32>, l: usize, r: usize) -> Option<Rc<RefCell<TreeNode>>> {
        let mid = (l+r) / 2;
        let mut node = TreeNode::new(nums[mid]);
        if mid > l {
            node.left = dfs(nums, l, mid-1);
        }
        if mid < r {
            node.right = dfs(nums, mid+1, r);
        }
        Some(Rc::new(RefCell::new(node)))
    }

    dfs(&nums, 0, nums.len() - 1)
}

#[derive(PartialEq, Eq, Clone, Debug)]
pub struct ListNode {
    pub val: i32,
    pub next: Option<Box<ListNode>>
  }
  
  impl ListNode {
    #[inline]
    fn new(val: i32) -> Self {
      ListNode {
        next: None,
        val
      }
    }
  }

pub fn sort_list(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    
    fn sort(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        let mut head = head;
        if head.is_none() || head.as_ref().unwrap().next.is_none() {
            return head;
        }
        let mut pre = & head;
        let mut step = 0;
        while pre.as_ref().unwrap().next.is_some() {
            pre = & pre.as_ref().unwrap().next;
            step += 1;
        }
        let mut cur = &mut head;
        for _ in 0..step/2-1 {
            cur = &mut cur.as_mut().unwrap().next;
        }
        let slow = cur.as_mut().unwrap().next.take();
        merge(sort(head),sort(slow))
    }

    fn merge(l: Option<Box<ListNode>>, r: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        match (l, r) {
            (None, Some(n)) | (Some(n), None) => Some(n),
            (Some(mut l), Some(mut r)) => {
                if l.val < r.val {
                    l.next = merge(l.next.take(), Some(r));
                    Some(l)
                } else {
                    r.next = merge(Some(l), r.next.take());
                    Some(r)
                }
            }
            (None, None) => None,
        }
    }
    sort(head)
}



pub fn merge_k_lists(lists: Vec<Option<Box<ListNode>>>) -> Option<Box<ListNode>> {
    use std::cmp;
    use std::collections::BinaryHeap;
    impl PartialOrd for ListNode {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.val.cmp(&other.val).reverse())
        }
    }
    
    impl Ord for ListNode {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.partial_cmp(other).unwrap()
        }
    }
    
    let mut heap = BinaryHeap::new();
    for n in lists {
        if n.is_some() {
            heap.push(n.unwrap());
        }
    }

    let mut dummy = Some(Box::new(ListNode::new(0)));
    let mut pre = &mut dummy;
    while let Some(mut node) = heap.pop() {
        let next_node = node.next.take();
        if next_node.is_some() {
            heap.push(next_node.unwrap());
        }
        pre.as_mut().unwrap().next = Some(Box::new(ListNode::new(node.val))); 
        pre = &mut pre.as_mut().unwrap().next;
    }

    dummy.unwrap().next
}

pub fn search_range(nums: Vec<i32>, target: i32) -> Vec<i32> {
    if nums.len() == 0 {
        return vec![-1, -1];
    }

    let mut l = 0;
    let mut r = nums.len() - 1;
    let mut first = -1;
    let mut last = -1;

    while l < r {
        let mid = (l+r) / 2;
        if nums[mid] < target {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    if nums[l] == target {
        first = l as i32;
    }
    l = 0;
    r = nums.len() - 1;
    while l < r {
        let mid = (l+r) / 2;
        if nums[mid] > target {
            r = mid - 1;
        } else {
            l = mid;
        }
    }
    if nums[l] == target {
        last = l as i32;
    }
    vec![first, last]
}


pub fn find_min(nums: Vec<i32>) -> i32 {
    if nums[0] < nums[nums.len()-1] {
        return nums[0];
    }
    let mut l = 0;
    let mut r = nums.len() - 1;
    while l < r {
        let mid = (l+r) / 2;
        if nums[mid] >= nums[0] {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    nums[l]
}

pub fn find_median_sorted_arrays(nums1: Vec<i32>, nums2: Vec<i32>) -> f64 {
    
    fn find_kth_number(nums1: &Vec<i32>, nums2: &Vec<i32>, i: usize, j: usize, k: usize) -> i32 {
        if nums1.len() - i > nums2.len() - j {
            return find_kth_number(nums2, nums1, j, i, k);
        }
        if nums1.len() == i {
            return nums2[j+k-1];
        }
        if k == 1 {
            return nums1[i].min(nums2[j]);
        }
        let si = nums1.len().min(i+k/2);
        let sj = j + k/2;
        if nums1[si-1] > nums2[sj-1] {
            return find_kth_number(nums1, nums2, i, sj, k - k/2);
        } else {
            return find_kth_number(nums1, nums2, si, j, k-si+i);
        }
    }

    let total = nums1.len() + nums2.len();
    if total % 2 == 0 {
        let left = find_kth_number(&nums1, &nums2, 0, 0, total/2);
        let right = find_kth_number(&nums1, &nums2, 0, 0, total/2+1);
        return (left + right) as f64 / 2 as f64;
    } else {
        return find_kth_number(&nums1, &nums2, 0, 0, total/2+1) as f64;
    }
}






