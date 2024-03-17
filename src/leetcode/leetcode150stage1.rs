
use std::{cell::{Ref, RefCell}, collections::{HashMap, HashSet, LinkedList}, rc::Rc};
use rand::seq::SliceRandom;

/// the first 66 questions in leetcode150
/// 数组、字符串、双指针、滑动窗口、矩阵、哈希表、区间、栈、链表。
/// 
/// 
pub fn alternating_subarray(nums: Vec<i32>) -> i32 {
    let mut res = -1;
    for i in 0..nums.len() {
        for j in i+1..nums.len() {
            let l = j-i+1;
            if nums[j]-nums[i] == (l-1) as i32 %2 {
                res = res.max(l as i32);
            } else {
                 break;
            }
        }
    }
    res
}


pub fn merge(nums1: &mut Vec<i32>, m: i32, nums2: &mut Vec<i32>, n: i32) {
    let mut m = m as usize;
    let mut n = n as usize;

    while m > 0 || n > 0 {
        if n == 0 || (n > 0 && m > 0 && nums1[m-1] > nums2[n-1]) {
            nums1[m+n-1] = nums1[m-1];
            m -= 1;
        } else {
            nums1[m+n-1] = nums2[n-1];
            n -= 1;
        }
    }
}

pub fn remove_element(nums: &mut Vec<i32>, val: i32) -> i32 {
    let mut index = 0_usize;
    for i in 0..nums.len() {
        if nums[i] != val {
            nums[index] = nums[i];
            index += 1;
        }
    }
    index as i32
}

pub fn remove_element_v2(nums: &mut Vec<i32>, val: i32) -> i32 {
    nums.retain(|x| *x != val);
    nums.len() as i32
}

pub fn remove_duplicates(nums: &mut Vec<i32>) -> i32 {
    let mut index = 0_usize;
    for i in 1..nums.len() {
        if nums[index] != nums[i] {
            index += 1;
            nums[index] = nums[i];
        }
    }
    index as i32 + 1
}

pub fn remove_duplicates_v2(nums: &mut Vec<i32>) -> i32 {
    if nums.len() < 3 {
        return nums.len() as i32;
    }
    let mut index = 2_usize;
    for i in 2..nums.len() {
        if nums[i] != nums[index-2] {
            nums[index] = nums[i];
            index += 1;
        }
    }
    index as i32
}

pub fn majority_element(nums: Vec<i32>) -> i32 {
    let mut h = HashMap::new();
    for i in nums {
        let count = h.entry(i).or_insert(0);
        *count += 1;
    }
    if let Some((key, _value)) = h.into_iter().max_by_key(|(_, k)| *k) {
        return key;
    } else {
        return 0;
    }
}

pub fn majority_element_v2(nums: Vec<i32>) -> i32 {
    let (mut candidate, mut count) = (nums[0], 1);
    for i in 1..nums.len() {
        if nums[i] == candidate {
            count += 1;
        } else if count > 0 {
            count -= 1;
        } else {
            (candidate, count) = (nums[i], 1);
        }
    }
    candidate
}

pub fn rotate(nums: &mut Vec<i32>, k: i32) {
    let mut k = k as usize;
    let l = nums.len();
    k %= l;
    let k1 = l - k;
    if k != 0 {
        for i in 0..(k1-1)/2 + 1 {
            nums.swap(i, k1-i-1)
        }
        for i in k1..(k-1)/2 + k1 + 1 {
            nums.swap(i, l-i-1+k1);
        }
        nums.reverse();
    } 
}

pub fn max_profit(prices: Vec<i32>) -> i32 {
    let (mut ret, mut mini_price) = (0, i32::MAX);
    for price in prices {
        ret = ret.max(price - mini_price);
        mini_price = mini_price.min(price);
    }
    ret
}

// dp[i][j] 表示第i天,持股状态为j时，最大的现金数量
// j = 0表示当天不持股， j = 1表示持股
pub fn max_profit_v2(prices: Vec<i32>) -> i32 {
    if prices.len() < 2 {
        return 0;
    }
    let mut dp = vec![vec![0; 2]; prices.len()];
    dp[0][0] = 0;
    dp[0][1] = -prices[0];
    for i in 1..prices.len() {
        dp[i][0] = dp[i-1][0].max(dp[i-1][1] + prices[i]);
        dp[i][1] = dp[i-1][1].max(dp[i-1][0] - prices[i]);
    }
    dp[prices.len()-1][0]
}

pub fn can_jump(nums: Vec<i32>) -> bool {
    let mut dist = 0;
    for i in 0..nums.len() {
        if i as i32 > dist {
            return false;
        }
        dist = dist.max(i as i32 +nums[i]);
    }
    true
}

pub fn jump(nums: Vec<i32>) -> i32 {
    let (mut res, mut start, mut end) = (0, 0_usize, nums[0] as usize + 1);
    while end < nums.len() {
        let mut p = 0_usize;
        for i in start..end {
            p = p.max(i+nums[i] as usize);
        }
        (start, end) = (end, p+1);
        res += 1;
    }
    res
}

pub fn h_index(citations: Vec<i32>) -> i32 {
    let mut citations = citations;
    citations.sort();
    let n = citations.len();
    let (mut l, mut r) = (0, citations.len()-1);
    while l < r {
        let mid = (l+r+1) / 2;
        if citations[n - mid] >= mid as i32 {
            l = mid;
        } else {
            r = mid - 1;
        }
    }
    l as i32
}

pub fn h_index_v2(citations: Vec<i32>) -> i32 {
    let mut citations = citations;
    citations.sort();
    for (i, v) in citations.iter().enumerate() {
        if *v as usize >= citations.len() - i {
            return (citations.len() - i) as i32;
        }
    }
    0
}


struct RandomizedSet {
    nums: Vec<i32>,
    indices: HashMap<i32, usize>,
}


/**
 * `&self` means the method takes an immutable reference.
 * If you need a mutable reference, change it to `&mut self` instead.
 */
impl RandomizedSet {

    fn new() -> Self {
       Self{
        nums: Vec::new(),
        indices: HashMap::new(),
       }
    }
    
    fn insert(&mut self, val: i32) -> bool {
        if self.indices.contains_key(&val) {
            return false;
        }
        self.indices.insert(val, self.nums.len());
        self.nums.push(val);
        true
    }
    
    fn remove(&mut self, val: i32) -> bool {
        if !self.indices.contains_key(&val) {
            return false;
        }
        let id = self.indices.get(&val).copied().unwrap();
        let last = self.nums.len() - 1;
        self.nums.swap(id, last);
        self.indices.insert(self.nums[id], id);
        self.nums.remove(last);
        self.indices.remove(&val);
        true
    }
    
    fn get_random(&self) -> i32 {
        let mut rng = rand::thread_rng();
        self.nums.choose(&mut rng).copied().unwrap()
    }
}

pub fn product_except_self(nums: Vec<i32>) -> Vec<i32> {
    let mut res = vec![0; nums.len()];
    let mut t = 1;
    for (i, value) in res.iter_mut().enumerate() {
        *value = t;
        t *= nums[i];
    }
    t = 1;
    for (i, value) in res.iter_mut().enumerate().rev() {
        *value *= t;
        t *= nums[i];
    }
    res
}

pub fn can_complete_circuit(gas: Vec<i32>, cost: Vec<i32>) -> i32 {
    let (mut i, n) = (0, gas.len());
    while i < n  {
        let (mut j, mut gas_left) = (0, 0);
        while j < n {
            let k = (i+j) % n;
            gas_left += gas[k] - cost[k];
            if gas_left < 0 {
                break;
            }
            j += 1;
        }
        if j >= n {
            return i as i32;
        }
        i += j + 1;
    }
    -1
}

pub fn candy(ratings: Vec<i32>) -> i32 {
    let mut res = vec![0; ratings.len()];
    let mut cur = 1;
    for (i, value) in res.iter_mut().enumerate() {
        if i > 0 && ratings[i] > ratings[i-1] {
           cur += 1;
        } else {
            cur = 1;
        }
        *value = cur;
    }
    let mut r = 0;
    for i in (0..=ratings.len()-1).rev() {
        if i < ratings.len()-1 && ratings[i] > ratings[i+1] {
            r += 1;
        } else {
            r = 1;
        }
        res[i] = res[i].max(r);
    }
    res.iter().sum()
}

pub fn trap(height: Vec<i32>) -> i32 {
    let (max_index, _) = height.iter().enumerate().max_by_key(|(_, &value)| value).unwrap();
    let (mut water, mut left, mut right) = (0, 0, 0);
    for i in 0..max_index {
        if height[i] > left {
            left = height[i];
        } else {
            water += left - height[i];
        }
    }
    for i in (max_index..height.len()).rev() {
        if height[i] > right {
            right = height[i];
        } else {
            water += right - height[i];
        }
    }
    water
}

pub fn roman_to_int(s: String) -> i32 {
    if s.is_empty() {
        return 0;
    }
    let sub_vec = vec![
        ("CM", 200), ("CD", 200),
        ("XC", 20), ("XL", 20),
        ("IX", 2), ("IV", 2),
    ];
    let sub_map :HashMap<_, _> = sub_vec.into_iter().collect();
    let add_vec = vec![
        ('M', 1000), ('D', 500), ('C', 100),
        ('L', 50), ('X', 10), ('V', 5), ('I', 1),
    ];
    let add_map :HashMap<_, _> = add_vec.into_iter().collect();
    let mut res = 0;
    for (k, v) in sub_map {
        if s.contains(k) {
            res -= v;
        }
    }
    for c in s.chars() {
        if add_map.contains_key(&c) {
            let add_value = add_map.get(&c).copied().unwrap();
            res += add_value;
        }
    }
    res
}

pub fn int_to_roman(num: i32) -> String {
    let values = vec![1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1];
    let reps = vec!["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"];
    let mut res = String::new();
    let mut num = num;
    for i in 0..13 {
        while num >= values[i] {
            num -= values[i];
            res += reps[i];
        }
    }
    res
}

pub fn length_of_last_word(s: String) -> i32 {
    let string_vec :Vec<String> = s.split_whitespace().map(String::from).collect();
    if string_vec.len() == 0 {
        return 0;
    }
    string_vec.last().unwrap().len() as i32
}

pub fn longest_common_prefix(strs: Vec<String>) -> String {
    let mut strs = strs;
    if strs.len() == 1 {
        return strs[0].clone();
    }
    strs.sort();
    let (first, last) = (strs[0].clone(), strs[strs.len()-1].clone());
    let mut commom_prefix = String::new();
    for (c1, c2) in first.chars().zip(last.chars()) {
        if c1 != c2 {
            break;
        }
        commom_prefix.push(c1);
    }
    commom_prefix

}

pub fn reverse_words(s: String) -> String {
    let string_vec :Vec<String> = s.split_whitespace().map(String::from).rev().collect();
    string_vec.join(" ")
}

pub fn convert(s: String, num_rows: i32) -> String {
    let mut matrix = vec![Vec::<char>::new(); num_rows as usize];
    let mut i = 0;
    let s_chars :Vec<char> = s.chars().collect();
    while i < s_chars.len() {
        for idx in 0..num_rows as usize {
            if i < s_chars.len() {
                matrix[idx].push(s_chars[i]);
                i += 1;
            }
        }
        let end = 1.max(num_rows-2);
        for idx in (1..=end as usize).rev() {
            if i < s_chars.len() {
                matrix[idx].push(s_chars[i]);
                i += 1;
            }
        }
    }
    let res_chars = matrix.concat();
    res_chars.iter().collect()
}

pub fn str_str(haystack: String, needle: String) -> i32 {
    if let Some(index) = haystack.find(&needle) {
        return index as i32
    }
    -1
}

pub fn full_justify(words: Vec<String>, max_width: i32) -> Vec<String> {
    let mut res = Vec::new();
    let mut i = 0;
    while i < words.len() {
        let (mut j, mut s, mut rs) = (i+1, words[i].len(), words[i].len());
        while j < words.len() && s+1+words[j].len() <= max_width as usize {
            s += words[j].len() + 1;
            rs += words[j].len();
            j += 1;
        }
        rs = max_width as usize - rs;
        let mut line = words[i].clone();
        if j == words.len() {
            for k in i+1..j {
                line += " ";
                line += &words[k];
            }
            line += &" ".repeat(max_width as usize - line.len());
        } else if j - i == 1 {
            line += &" ".repeat(max_width as usize - line.len());
        } else {
            let (base, rem) = (rs / (j-i-1), rs % (j-i-1));
            i += 1;
            let mut k = 0;
            while i < j {
                let mut carry = 0;
                if k < rem {
                    carry = 1;
                }
                line += &" ".repeat(base + carry);
                line += &words[i];
                k += 1;
                i += 1;
            }
        }
        i = j;
        res.push(line);
    }
    res
}

/// double pointer
pub fn is_palindrome(s: String) -> bool {
    let s :String = s.chars()
                        .filter(|c| c.is_alphanumeric())
                        .map(|c| c.to_lowercase())
                        .flatten()
                        .collect();
    s == s.chars().rev().collect::<String>()
}

pub fn is_subsequence(s: String, t: String) -> bool {
    let vec_s :Vec<char> = s.chars().collect();
    let mut s_index = 0;
    for (_, t_c) in t.chars().enumerate() {
        if s_index == s.len() {
            return true
        }
        if t_c == vec_s[s_index] {
            s_index += 1;
        }
    }
    s_index == s.len()
}


pub fn two_sum(numbers: Vec<i32>, target: i32) -> Vec<i32> {
    let (mut i, mut j) = (0, numbers.len() - 1);
    while i < j {
        let sum = numbers[i] + numbers[j];
        if sum < target {
            i += 1;
        } else if sum > target {
            j -= 1;
        } else {
            return vec![i as i32 + 1, j as i32 + 1];
        }
    }
    vec![-1, -1]
}

pub fn max_area(height: Vec<i32>) -> i32 {
    let (mut ret, mut left, mut right) = (0, 0, height.len() - 1);
    while left < right {
        ret = ret.max((right-left) as i32 * height[left].min(height[right]));
        if height[left] >= height[right] {
            right -= 1;
        } else {
            left += 1;
        }
    }
    ret
}

pub fn three_sum(nums: Vec<i32>) -> Vec<Vec<i32>> {
    let mut ret = Vec::new();
    let mut nums = nums;
    nums.sort();
    for i in 0..nums.len()-2 {
        if nums[i] > 0 {
            break;
        }
        let mut start = i + 1;
        if i > 0 && nums[i] == nums[i-1] {
            continue;
        }
        let mut end = nums.len() - 1;
        while start < end {
            if nums[i] + nums[start] + nums[end] < 0 {
                start += 1;
                while nums[start] == nums[start-1] && start < end {
                    start += 1;
                }
            } else if nums[i] + nums[start] + nums[end] > 0 {
                end -= 1;
                while nums[end] == nums[end+1] && start < end {
                    end -= 1;
                }
            } else {
                ret.push(vec![nums[i], nums[start], nums[end]]);
                start += 1;
                end -= 1;
                while nums[start] == nums[start-1] && nums[end] == nums[end+1] && start < end {
                    start += 1;
                    end -= 1;
                }
            }
        }
    }
    ret
}

/// slide window
pub fn min_sub_array_len(target: i32, nums: Vec<i32>) -> i32 {
    let mut ans = i32::MAX;
    let mut sums = vec![0; nums.len()+1];
    for i in 1..=nums.len() {
        sums[i] = sums[i-1] + nums[i-1];
    }
    for i in 1..=nums.len() {
        let cur_target = target + sums[i-1];
        let search_ret = sums.binary_search_by(|x| x.cmp(&cur_target));
        let bound;
        match search_ret {
            Ok(x) => bound = x,
            Err(y) => bound = y,
        }
        if bound <= nums.len() {
            ans = ans.min((bound - i) as i32 + 1);
        }
    }
    if ans == i32::MAX {
        return 0;
    }
    ans
}

pub fn length_of_longest_substring(s: String) -> i32 {
    let (mut left, mut ret) = (0, 0);
    let mut visited = HashMap::new();
    let s_chars :Vec<char> = s.chars().collect();
    for right in 0..s_chars.len() {
        if let Some(&index) = visited.get(&s_chars[right]) {
            if index >= left {
                left = index + 1;
            }
        }
        visited.insert(s_chars[right], right);
        ret = ret.max((right - left + 1) as i32);
    }
    ret
}

pub fn find_substring(s: String, words: Vec<String>) -> Vec<i32> {
    let (n, m, w) = (s.len() as i32, words.len() as i32, words[0].len() as i32);
    let mut ret = Vec::new();
    let mut target :HashMap<String, i32> = HashMap::new();
    let mut test_map :HashMap<String, i32> = HashMap::new();
    for (_, v) in words.iter().enumerate() {
        let count = target.entry(v.clone()).or_insert(0);
        *count += 1;
    }
    for i in 0..w {
        test_map.clear();
        let (mut sumn, mut left, mut right) = (0, i as i32, i as i32);
        while right + w <= n {
            let mut word;
            if right - m*w >= left {
                word = &s[(right-m*w) as usize..(right-m*w+w) as usize];
                if let Some(&test_value) = test_map.get(word) {
                    if let Some(&target_value) = target.get(word) {
                        if test_value == target_value {
                            sumn -= test_value
                        }
                    }
                }
                let count = test_map.entry(word.to_string()).or_insert(0);
                *count -= 1;
                if let Some(&test_value) = test_map.get(word) {
                    if let Some(&target_value) = target.get(word) {
                        if test_value == target_value {
                            sumn += test_value
                        }
                    }
                }
            }
            word = &s[right as usize..(right+w) as usize];
            if !target.contains_key(word) {
                test_map.clear();
                sumn = 0;
                left = right + w;
            } else {
                if let Some(&test_value) = test_map.get(word) {
                    if let Some(&target_value) = target.get(word) {
                        if test_value == target_value {
                            sumn -= test_value
                        }
                    }
                }
                let count = test_map.entry(word.to_string()).or_insert(0);
                *count += 1;
                if let Some(&test_value) = test_map.get(word) {
                    if let Some(&target_value) = target.get(word) {
                        if test_value == target_value {
                            sumn += test_value
                        }
                    }
                }
            }
            if sumn == m {
                ret.push((right-m*w+w) as i32);
            }
            right += w;
        }
    }
    ret
}

pub fn min_window(s: String, t: String) -> String {
    let mut target :HashMap<char, i32> = HashMap::new();
    let mut source :HashMap<char, i32> = HashMap::new();
    let mut ans = "".to_string();
    for c in t.chars() {
        let count = target.entry(c).or_insert(0);
        *count += 1;
    }
    println!("target, {:?}", target);
    let s_chars :Vec<char> = s.chars().collect();
    let (mut statisfy, total, mut j) = (0, target.len(), 0);
    for (i, c) in s.chars().enumerate() {
        let source_count = source.entry(c).or_insert(0);
        *source_count += 1;
        if let Some(&source_value) = source.get(&c) {
            if let Some(&target_value) = target.get(&c) {
                if source_value == target_value {
                    statisfy += 1;
                }
            }
        }
        println!("statisfy: {:?}", statisfy);
        println!("source: {:?}", source);
        while j < s.len() && 
            source.entry(s_chars[j]).or_default() > target.entry(s_chars[j]).or_default()
        {
            println!("in while {:?}", j);
            *source.get_mut(&s_chars[j]).unwrap() -= 1;
            j += 1;
        }
        if statisfy == total && (ans.len() == 0 || i-j+1 < ans.len()) {
            ans = (&s[j..i+1]).to_string();
        }
    }
    ans
}

pub fn is_valid_sudoku(board: Vec<Vec<char>>) -> bool {
    let mut row_buf = vec![vec![false; 9]; 9];
    let mut col_buf = vec![vec![false; 9]; 9];
    let mut box_buf = vec![vec![false; 9]; 9];

    for i in 0..9 {
        for j in 0..9 {
            if board[i][j] != '.' {
                let num = board[i][j].to_digit(10).unwrap() as usize - 1;
                if row_buf[i][num] || col_buf[j][num] || box_buf[i/3*3+j/3][num] {
                    return  false;
                }
                row_buf[i][num] = true;
                col_buf[j][num] = true;
                box_buf[i/3*3+j/3][num] = true;
            }
        }
    }
    true
}

pub fn spiral_order(matrix: Vec<Vec<i32>>) -> Vec<i32> {
    let (m, n) = (matrix.len(), matrix[0].len());
    let mut res = vec![0; m*n];
    let mut visited = vec![vec![false; n]; m];
    let dx: Vec<i32> = vec![-1, 0, 1, 0];
    let dy: Vec<i32> = vec![0, 1, 0, -1];
    let (mut x, mut y, mut d) = (0_i32, 0_i32, 1);
    for i in 0..m*n {
        res[i] = matrix[x as usize][y as usize];
        visited[x as usize][y as usize] = true;
        let (mut next_x, mut next_y) = ((x+dx[d])  , y+dy[d]);
        if next_x < 0 || next_x >= m as i32 || next_y < 0 || next_y >= n as i32 || visited[next_x as usize][next_y as usize] {
            d = (d+1) % 4;
            (next_x, next_y) = (x+dx[d], y+dy[d]);
        }
        (x, y) = (next_x, next_y);
    }
    res
}

pub fn rotate_2(matrix: &mut Vec<Vec<i32>>) {
    let n = matrix.len();
    for i in 0..n {
        for j in i+1..n {
            let temp = matrix[i][j];
            matrix[i][j] = matrix[j][i];
            matrix[j][i] = temp;
        }
    }
    for i in 0..n {
        for j in 0..n/2 {
            let temp = matrix[i][j];
            matrix[i][j] = matrix[i][n-j-1];
            matrix[i][n-j-1] = temp;
        }
    }
}

pub fn set_zeroes(matrix: &mut Vec<Vec<i32>>) {
    let mut set_row = HashSet::new();
    let mut set_col = HashSet::new();
    for i in 0..matrix.len(){
        for j in 0..matrix[0].len(){
            if matrix[i][j] == 0 {
                set_row.insert(i);
                set_col.insert(j);
            }
        }
    }
    for i in set_row{
        for j in 0..matrix[0].len(){
            matrix[i][j] = 0;
        }
    }
    for j in set_col{
        for i in 0..matrix.len(){
            matrix[i][j] = 0;
        }
    }
}

pub fn game_of_life(board: &mut Vec<Vec<i32>>) {
    let (m, n) = (board.len() as i32, board[0].len() as i32);
    let dx = vec![-1, -1, -1, 0, 1, 1, 1, 0];
    let dy = vec![-1, 0, 1, 1, 1, 0, -1 ,-1];
    for i in 0..m {
        for j in 0..n {
            let mut count = 0;
            for k in 0..8 {
                let (x, y) = (i + dx[k], j+ dy[k]);
                if x >= 0 && x < m && y >= 0 && y < n {
                    if board[x as usize][y as usize] == 1 || board[x as usize][y as usize] == 2 {
                        count += 1;
                    }
                }
            }
            if  board[i as usize][j as usize] == 1 && (count < 2 || count > 3) {
                board[i as usize][j as usize] = 2; // 2 表示本来是活细胞，但是已经被判死刑了
            } else if board[i as usize][j as usize] == 0 && count == 3 {
                board[i as usize][j as usize] = 3; // 3 表示本来是死细胞，现在被救活了
            }
        }
    }
    for i in 0..m {
        for j in 0..n {
            board[i as usize][j as usize] %= 2;
        }
    }
}

pub fn can_construct(ransom_note: String, magazine: String) -> bool {
    let mut map_source = HashMap::new();
    for c in magazine.chars() {
        let count = map_source.entry(c).or_insert(0);
        *count += 1;
    }
    
    for c in ransom_note.chars() {
        let count = map_source.entry(c).or_default();
        if *count == 0 {
            return false;
        }
        *count -= 1;
    }
    true
}

pub fn is_isomorphic(s: String, t: String) -> bool {
    let mut dic = HashMap::new();
    let mut set = HashSet::new();
    for (s_c, t_c) in s.chars().zip(t.chars()) {
        if !dic.contains_key(&s_c) {
            if  set.contains(&t_c) {
                return false;
            }
            dic.insert(s_c, t_c);
            set.insert(t_c);
        }
        if let Some(&c) = dic.get(&s_c) {
            if c != t_c {
                return false;
            }
        }
    }
    true
}

pub fn word_pattern(pattern: String, s: String) -> bool {
    if pattern.len() != s.split_whitespace().count() {
        return false;
    }
    let mut dic = HashMap::new();
    let mut set = HashSet::new();
    for (p_c, s_s) in pattern.chars().zip(s.split_whitespace()) {
        println!("{:?}, {:?}", p_c, s_s);
        if !dic.contains_key(&p_c) {
            if set.contains(s_s) {
                return false;
            }
            dic.insert(p_c, s_s);
            set.insert(s_s);
        }
        if let Some(&p_s) = dic.get(&p_c) {
            if p_s != s_s {
                return false;
            }
        }
    }
    true
}

pub fn is_anagram(s: String, t: String) -> bool {
    let mut s_chars :Vec<char> = s.chars().collect();
    let mut t_chars :Vec<char> = t.chars().collect();
    s_chars.sort();
    t_chars.sort();

    s_chars == t_chars

}

pub fn group_anagrams(strs: Vec<String>) -> Vec<Vec<String>> {
    let mut dic: HashMap<String, Vec<String>> = HashMap::new();
    let mut ret = Vec::new();
    for s in strs {
        let sorted_s = sort_str(&s);
        let entry = dic.entry(sorted_s).or_insert(Vec::new());
        entry.push(s.clone());
    }
    for (_, v) in dic {
        ret.push(v);
    }
    ret
}

fn sort_str(s: &str) -> String {
    let mut s_chars :Vec<char> = s.chars().collect();
    s_chars.sort();

    s_chars.into_iter().collect()
} 

pub fn two_sum_2(nums: Vec<i32>, target: i32) -> Vec<i32> {
    let mut m = HashMap::new();
    for (i, v) in nums.iter().enumerate() {
        let index = target - v;
        if m.contains_key(&index) {
            return vec![*m.get(&index).unwrap(), i as i32];
        }
        m.insert(*v, i as i32);
    }
    vec![]
}

pub fn is_happy(n: i32) -> bool {
    let mut visited = HashSet::new();
    let mut n = n;
    while n != 1 {
       visited.insert(n);
       n = get_number(n);
       if visited.contains(&n) {
        return false;
       }
    }
    true
}

fn get_number(n: i32) -> i32 {
    let (mut ret, mut tmp) = (0, n);
    while tmp != 0 {
        let remainder = tmp % 10;
        ret += remainder * remainder;
        tmp /= 10;
    }
    ret
}

pub fn contains_nearby_duplicate(nums: Vec<i32>, k: i32) -> bool {
    let mut m = HashMap::new();
    for (i, v) in nums.into_iter().enumerate() {
        if let Some(index) = m.insert(v, i as i32) {
            if (i as i32 - index).abs() <= k {
                return true;
            }
        }
    }
    false
}

pub fn longest_consecutive(nums: Vec<i32>) -> i32 {
    let set: HashSet<i32> = nums.clone().into_iter().collect();
    let mut ret = 0;
    for num in nums {
        if !set.contains(&(num-1)) {
            let (mut current, mut current_ret) = (num, 0);
            while set.contains(&current) {
                current += 1;
                current_ret += 1;
            }
            ret = ret.max(current_ret);
        }
    }
    ret

}

pub fn summary_ranges(nums: Vec<i32>) -> Vec<String> {
    let mut res = Vec::new();
    if nums.len() == 0 {
        return res;
    }
    let (mut left, mut right) = (nums[0], nums[0]);
    for &num in &nums[1..] {
        if num > right+1 {
            res.push(merge_range(left, right));
            left = num;
            right = num;
        } else {
            right += 1;
        }
    }
    res.push(merge_range(left, right));
    res
}

fn merge_range(a: i32, b: i32) -> String {
    if a == b {
        a.to_string()
    } else {
        vec![a.to_string(), b.to_string()].join("->")
    }
}

pub fn merge_2(intervals: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let mut intervals = intervals;
    intervals.sort_by(|a, b| {
        if a[0] != b[0] {
            a[0].cmp(&b[0])
        } else {
            a[1].cmp(&b[1])
        }
    });
    let mut res = Vec::new();
    res.push(intervals[0].clone());
    let mut res_index = 0;
    for i in 1..intervals.len() {
        if intervals[i][0] > res[res_index][1] {
            res_index += 1;
            res.push(intervals[i].clone());
        } else {
            res[res_index][1] = res[res_index][1].max(intervals[i][1]);
        }
    }
    res
}

pub fn insert(intervals: Vec<Vec<i32>>, new_interval: Vec<i32>) -> Vec<Vec<i32>> {
    let mut intervals = intervals;
    intervals.push(new_interval);
    merge_2(intervals)
}

pub fn find_min_arrow_shots(points: Vec<Vec<i32>>) -> i32 {
    let mut points = points;
    points.sort_by_key(|a| a[1]);
    let (mut res, mut right) = (0, i32::MIN);
    if points[0][0] == right {
        res += 1;
    }
    for point in points {
        if point[0] > right {
            right = point[1];
            res += 1;
        }
    }
    res
}

pub fn is_valid(s: String) -> bool {
    let mut stack = vec![];
    for c in s.chars() {
        match c {
            '(' | '{' | '[' => {
                stack.push(c)
            },
            ')' | '}' | ']' => {
                if let Some(cc) = stack.pop() {
                    let correct = match c {
                        ')' => cc == '(',
                        '}' => cc == '{',
                        ']' => cc == '[',
                        _ => true
                    };
                    if !correct {
                        return false;
                    }
                } else {
                    return false;
                }
            }
            _ => {}
        }
    }
    stack.is_empty()
}

pub fn simplify_path(path: String) -> String {
    let mut path = path;
    path += "/";
    let mut res = vec![];
    let mut s = "".to_string();
    for c in path.chars() {
        let x = c.to_string();
        if res.len() == 0 {
            res.push(x);
        } else if x != "/" {
            s += &x;
        } else {
            if s == ".." {
                if res.len() > 1 {
                    res.pop();
                    while let Some(cc) = res.last() {
                        if cc == "/" {
                            break;
                        } else {
                            res.pop();
                        }
                    }
                }
            } else if s != "." && !s.is_empty() {
                res.push(s);
                res.push("/".to_string());
            }
            s = "".to_string();
        }
    }
    if res.len() > 1 {
        res.pop();
    }
    res.join("")
}

struct MinStack {
    stack: Vec<i32>,
    min_stack: Vec<i32>,
}

impl MinStack {

    fn new() -> Self {
        Self {
            stack: vec![],
            min_stack: vec![i32::MAX],
        }
    }
    
    fn push(&mut self, val: i32) {
        self.stack.push(val);
        let min_val = self.min_stack.last().unwrap();
        self.min_stack.push(val.min(*min_val));
    }
    
    fn pop(&mut self) {
        self.stack.pop();
        self.min_stack.pop();
    }
    
    fn top(&self) -> i32 {
        *self.stack.last().unwrap()
    }
    
    fn get_min(&self) -> i32 {
        *self.min_stack.last().unwrap()
    }
}

pub fn eval_rpn(tokens: Vec<String>) -> i32 {
    let mut stack = vec![];

    for token in tokens {
       if let Ok(val) = token.parse::<i32>() {
        stack.push(val);
       } else {
        let num2 = stack.pop().unwrap();
        let num1 = stack.pop().unwrap();
        match token.as_str() {
            "+" => stack.push(num1 + num2),
            "-" => stack.push(num1 - num2),
            "*" => stack.push(num1 * num2),
            "/" => stack.push(num1 / num2),
            _ => {}

        }
       }
    }
    stack[0]
}


pub fn calculate(s: String) -> i32 {
    let mut op = vec![1];
    let mut sign = 1;
    let mut res = 0;

    let mut num_chars = vec![];
    let s_chars :Vec<char> = s.chars().into_iter().collect();
    for i in 0..s_chars.len() {
        let c = s_chars[i];
        match c {
            ' ' => {},
            '(' => op.push(sign),
            '+' => {
                sign = *op.last().unwrap()
            }
            '-' => sign = *op.last().unwrap() * -1,
            ')' => {
                op.pop();
            },
            _ => {
                num_chars.push(c);
                if i == s_chars.len() - 1 || !s_chars[i+1].is_numeric() {
                    let num_val = num_chars.iter().collect::<String>().parse::<i32>().unwrap();
                    res += sign * num_val;
                    num_chars.clear();

                }
            }
        }
    }
    res
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

pub fn add_two_numbers(l1: Option<Box<ListNode>>, l2: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    let mut dummy = Some(Box::new(ListNode::new(0)));
    
    let mut carry = 0;
    let mut l1 = l1;
    let mut l2 = l2;
    let mut cur = &mut dummy;
    while l1.is_some() || l2.is_some() || carry != 0 {
        let n1;
        let n2;
        if let Some(node1) = l1 {
            n1 = node1.val;
            l1 = node1.next;
        } else {
            n1 = 0;
        }
        if let Some(node2) = l2 {
            n2 = node2.val;
            l2 = node2.next;
        } else {
            n2 = 0;
        }

        let new_val = (n1+n2+carry) % 10;
        let new_node = Box::new(ListNode::new(new_val));
        cur.as_mut().unwrap().next = Some(new_node);
        cur = &mut cur.as_mut().unwrap().next;
        carry = (n1+n2+carry) / 10;
    } 
    dummy.unwrap().next
}


pub fn merge_two_lists(list1: Option<Box<ListNode>>, list2: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    return match (list1, list2) {
        (Some(mut cur1), Some(mut cur2)) => {
            if cur1.val <= cur2.val {
                let next = cur1.next.take();
                cur1.next = merge_two_lists(next, Some(cur2));
                return Some(cur1);
            } else {
                let next = cur2.next.take();
                cur2.next = merge_two_lists(next, Some(cur1));
                return Some(cur2);
            }
        },
        (x, y) => x.or(y),
    }
}


pub fn reverse_between(head: Option<Box<ListNode>>, left: i32, right: i32) -> Option<Box<ListNode>> {
    let mut dummy = Some(Box::new(ListNode{val:0, next: head}));
    let mut pre = &mut dummy;
    for _ in 1..left {
        pre = &mut pre.as_mut().unwrap().next;
    }
    let cur = &mut pre.as_mut().unwrap().next.take();

    for _ in 0..right-left {
        let mut cur_next =  cur.as_mut().unwrap().next.take();
        cur.as_mut().unwrap().next = cur_next.as_mut().unwrap().next.take();
        cur_next.as_mut().unwrap().next = pre.as_mut().unwrap().next.take();
        pre.as_mut().unwrap().next = cur_next;
    }

    while pre.as_ref().unwrap().next.is_some() {
        pre = &mut pre.as_mut().unwrap().next
    }
    pre.as_mut().unwrap().next = cur.take();

    dummy.unwrap().next
}


pub fn reverse_k_group(head: Option<Box<ListNode>>, k: i32) -> Option<Box<ListNode>> {
    let mut dummy = Some(Box::new(ListNode{val: 0, next: head}));
    let mut pre = &mut dummy;
    while  pre.as_ref().unwrap().next.is_some() {
        let mut pre_check = &mut pre.clone();
        let mut next_num_check = 0;
        while pre_check.as_ref().unwrap().next.is_some() {
            pre_check = &mut pre_check.as_mut().unwrap().next;
            next_num_check += 1;
            if next_num_check == k {
                break;
            }
        }
        if next_num_check < k {
            break;
        }
        let cur = &mut pre.as_mut().unwrap().next.take();
        for _ in 0..k-1 {
            let mut cur_next = cur.as_mut().unwrap().next.take();
            if cur_next.is_none() {
                break;
            }
            cur.as_mut().unwrap().next = cur_next.as_mut().unwrap().next.take();
            cur_next.as_mut().unwrap().next = pre.as_mut().unwrap().next.take();
            pre.as_mut().unwrap().next = cur_next;
        }
        while pre.as_ref().unwrap().next.is_some() {
            pre = &mut pre.as_mut().unwrap().next
        }
        pre.as_mut().unwrap().next = cur.take();
        pre = &mut pre.as_mut().unwrap().next
    }
    dummy.unwrap().next
}

pub fn remove_nth_from_end(head: Option<Box<ListNode>>, n: i32) -> Option<Box<ListNode>> {
    let mut dummy = Some(Box::new(ListNode{val: 0, next: head}));
    let mut pre = & dummy;
    let mut step = 0;
    for _ in 0..n {
        pre = & pre.as_ref().unwrap().next;
    }
    while pre.as_ref().unwrap().next.is_some() {
        pre = & pre.as_ref().unwrap().next;
        step += 1;
    }
    let mut cur = &mut dummy;
    for _ in 0..step {
        cur = &mut cur.as_mut().unwrap().next;
    }
    let mut cur_next = cur.as_mut().unwrap().next.take();
    cur.as_mut().unwrap().next = cur_next.as_mut().unwrap().next.take();
    dummy.unwrap().next
}

pub fn delete_duplicates(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    let mut head = head;
    let mut dummy = Some(Box::new(ListNode::new(101)));
    let mut pre = &mut dummy;
    let mut last_val = 101;
    while let Some(mut cur) = head {
        head = cur.next.take();
        if (head.is_some() && head.as_ref().unwrap().val == cur.val) || cur.val == last_val {
            last_val = cur.val;
        } else {
            last_val = cur.val;
            pre.as_mut().unwrap().next = Some(cur);
            pre = &mut pre.as_mut().unwrap().next;
        }
    }
    dummy.unwrap().next
}

pub fn rotate_right(head: Option<Box<ListNode>>, k: i32) -> Option<Box<ListNode>> {
    if head.is_none() || head.as_ref().unwrap().next.is_none() || k == 0 {
        return head;
    }
    let mut k = k;
    let mut length = 0;
    let mut pre_check = & head;
    while pre_check.as_ref().unwrap().next.is_some() {
        pre_check = & pre_check.as_ref().unwrap().next;
        length += 1;
    }
    k %= length;
    if k == 0 {
        return head;
    } 
    println!("k, {:?}", k);
    let mut res = Some(Box::new(ListNode::new(0)));
    let mut dummy = Some(Box::new(ListNode{val: 0, next: head}));
    let mut pre = & dummy;
    let mut step = 0;
    for _ in 0..k {
        pre = & pre.as_ref().unwrap().next;
    }
    while pre.as_ref().unwrap().next.is_some() {
        pre = & pre.as_ref().unwrap().next;
        step += 1;
    }

    let mut cur = &mut dummy;
    for _ in 0..step {
        cur = &mut cur.as_mut().unwrap().next;
    }
    res.as_mut().unwrap().next = cur.as_mut().unwrap().next.take();
    let old_pre = &mut dummy;
    let mut res_cur = &mut res;
    while res_cur.as_ref().unwrap().next.is_some() {
        res_cur = &mut res_cur.as_mut().unwrap().next;
    }
    res_cur.as_mut().unwrap().next = old_pre.as_mut().unwrap().next.take();

    res.unwrap().next
}


pub fn partition(head: Option<Box<ListNode>>, x: i32) -> Option<Box<ListNode>> {
    let (mut left, mut right) = (None, None);
    let mut left_p = &mut left;
    let mut right_p = &mut right;
    
    let mut p = head;
    while let Some(mut cur) = p {
        p = cur.next.take();
        if cur.val < x {
            left_p = &mut left_p.insert(cur).next;
        } else {
            right_p = &mut right_p.insert(cur).next;
        }
    }
    *left_p = right;

    left
}

#[derive(Debug)]
struct LruNode {
    key: i32,
    value: i32,
    pre: Option<Rc<RefCell<LruNode>>>,
    next:  Option<Rc<RefCell<LruNode>>>,
}

impl LruNode {
    fn new(key: i32, value: i32) -> Rc<RefCell<LruNode>> {
        Rc::new(RefCell::new(LruNode{
            key,
            value,
            pre: None,
            next: None,
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
        dummy.borrow_mut().pre = Some(Rc::clone(&dummy));
        dummy.borrow_mut().next = Some(Rc::clone(&dummy));
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
            let last_node = self.dummy.borrow().pre.clone().unwrap();
            self.map.remove(&last_node.borrow().key);
            self.remove(last_node);
        }
    }

    fn remove(&self, x: Rc<RefCell<LruNode>>) {
        let pre = x.borrow().pre.clone().unwrap();
        let next = x.borrow().next.clone().unwrap();
        pre.borrow_mut().next = Some(Rc::clone(&next));
        next.borrow_mut().pre = Some(Rc::clone(&pre));
    }

    fn push_front(&mut self, x: Rc<RefCell<LruNode>>) {
        let next = self.dummy.borrow().next.clone();
        x.borrow_mut().next = next.clone();
        x.borrow_mut().pre = Some(Rc::clone(&self.dummy));
        self.dummy.borrow_mut().next = Some(Rc::clone(&x));
        next.unwrap().borrow_mut().pre = Some(x);
        
    }
}


#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_lru() {
       let cache = LRUCache::new(3);
       println!("count after creating  = {}", Rc::strong_count(&cache.dummy));
       println!("{:?}", &cache.dummy.borrow().key)
    }
}