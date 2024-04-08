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

