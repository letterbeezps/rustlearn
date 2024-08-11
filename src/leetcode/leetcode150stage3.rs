


pub fn find_kth_largest(nums: Vec<i32>, k: i32) -> i32 {

    fn partition(nums: &mut Vec<i32>, l: i32, r: i32, k: i32) -> i32 {
        if l == r {
            return nums[l as usize];
        }
        let mut i = l - 1;
        let mut j = r + 1;
        let x = nums[(l + r) as usize / 2];
        while i < j {
            i += 1;
            while nums[i as usize] > x {
                i += 1;
            }
            j -= 1;
            while nums[j as usize] < x {
                j -= 1;
            }
            if i < j {
                nums.swap(i as usize, j as usize);
            }
        }
        let s1 = j-l+1;
        if k <= s1 {
            return partition(nums, l, j, k);
        }
        return partition(nums, j+1, r, k-s1);
    }
    let r = nums.len() - 1;
    let mut nums = nums;
    return partition(&mut nums, 0, r as i32, k);
}

pub fn find_maximized_capital(k: i32, w: i32, profits: Vec<i32>, capital: Vec<i32>) -> i32 {
    use std::collections::BinaryHeap;
    let n = capital.len();
    let mut arr = capital.into_iter().zip(profits.into_iter()).collect::<Vec<(i32, i32)>>();
    arr.sort_by_key(|x| x.0);

    let mut heap = BinaryHeap::new();
    let mut index = 0;
    let mut w = w;
    for _ in 0..k {
        while index < n && arr[index].1 <= w {
            heap.push(arr[index].1);
            index += 1;
        }
        if heap.is_empty() {
            break;
        }
        w += heap.pop().unwrap();
    }
    w
}

pub fn k_smallest_pairs(nums1: Vec<i32>, nums2: Vec<i32>, k: i32) -> Vec<Vec<i32>> {
    use std::cmp;
    use std::collections::BinaryHeap;
    
    #[derive(PartialEq, Eq)]
    struct Pair (i32, i32, i32);
    
    impl PartialOrd for Pair {
        fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
            Some(self.0.cmp(&other.0).reverse())
        }
    }

    impl Ord for Pair {
        fn cmp(&self, other: &Self) -> cmp::Ordering {
            self.partial_cmp(other).unwrap()
        }
    }

    let mut heap = BinaryHeap::new();
    for i in 0..k as usize {
        if i >= nums1.len() {
            break;
        }
        heap.push(Pair(nums1[i] + nums2[0] ,i as i32, 0));
    }
    let mut ret = vec![];
    while heap.len() > 0 && ret.len() < k as usize {
        let data = heap.pop().unwrap();
        let (i, j) = (data.1 as usize, data.2 as usize);
        ret.push(vec![nums1[i], nums2[j]]);
        if j + 1 < nums2.len()  {
            heap.push(Pair(nums1[i] + nums2[j+1], data.1, data.2+1));
        }
    }
    ret
}

use std::cmp::Reverse;
use std::collections::BinaryHeap;
struct MedianFinder {
    heap_min: BinaryHeap<i32>,
    heap_max: BinaryHeap<Reverse<i32>>,
}

impl MedianFinder {

    fn new() -> Self {
        Self {
            heap_max: BinaryHeap::new(),
            heap_min: BinaryHeap::new(),
        }
    }
    
    fn add_num(&mut self, num: i32) {
        if self.heap_min.is_empty() || num <= *self.heap_min.peek().unwrap() {
            self.heap_min.push(num);
            if self.heap_max.len() + 1 < self.heap_min.len() {
                self.heap_max.push(Reverse(self.heap_min.pop().unwrap()));
            }
        } else {
            self.heap_max.push(Reverse(num));
            if self.heap_max.len() > self.heap_min.len() {
                self.heap_min.push(self.heap_max.pop().unwrap().0);
            }
        }
    }
    
    fn find_median(&self) -> f64 {
        if self.heap_min.len() > self.heap_max.len() {
            return *self.heap_min.peek().unwrap() as f64;
        }
        (*self.heap_min.peek().unwrap() + self.heap_max.peek().unwrap().0) as f64 / 2 as f64
    }
}

pub fn is_interleave(s1: String, s2: String, s3: String) -> bool {
    let (row, col) = (s1.len(), s2.len());
    if s3.len() != row+col {
        return false;
    }
    let mut dp = vec![vec![false; col+1]; row+1];
    dp[0][0] = true;
    let s_c1 = s1.chars().into_iter().collect::<Vec<char>>();
    let s_c2 = s2.chars().into_iter().collect::<Vec<char>>();
    let s_c3 = s3.chars().into_iter().collect::<Vec<char>>();
    for i in 0..row {
        if s_c1[i] == s_c3[i] {
            dp[i+1][0] = dp[i][0];
        }
    }
    for i in 0..col {
        if s_c2[i] == s_c3[i] {
            dp[0][i+1] = dp[0][i];
        }
    }
    for i in 1..=row {
        for j in 1..=col {
            match (dp[i][j-1], dp[i-1][j]) {
                (true, true) => {
                    dp[i][j] = (s_c3[i+j-1] == s_c1[i-1]) || (s_c3[i+j-1] == s_c2[j-1])
                }
                (true, false) => {
                    dp[i][j] = (s_c3[i+j-1] == s_c2[j-1])
                }
                (false, true) => {
                    dp[i][j] = (s_c3[i+j-1] == s_c1[i-1])
                }
                _ => {}
            }
        }
    }
    dp[row][col]
}

pub fn maximal_square(matrix: Vec<Vec<char>>) -> i32 {
    let (m, n) = (matrix.len(), matrix[0].len());
    let mut res = 0;
    let mut dp = vec![vec![0; n]; m];
    for i in 0..m {
        for j in 0..n {
            if matrix[i][j] == '0' {
                dp[i][j] = 0;
            } else {
                dp[i][j] = 1;
                if i > 0 && j > 0 {
                    dp[i][j] += dp[i-1][j-1].min(dp[i-1][j]).min(dp[i][j-1])
                }
                res = res.max(dp[i][j])
            }
        }
    }
    res * res
}

pub fn add_binary(a: String, b: String) -> String {
    let mut v_a = a.chars().map(|c| c.to_digit(10).unwrap() as i32).collect::<Vec<i32>>();
    let mut v_b = b.chars().map(|c| c.to_digit(10).unwrap() as i32).collect::<Vec<i32>>();
    
    let mut res = vec![];
    let mut carry = 0;
    while !v_a.is_empty() || !v_b.is_empty() {
        let (mut na, mut nb) = (0, 0);
        match (v_a.pop(), v_b.pop()) {
            (Some(x), Some(y)) => {
                na = x;
                nb = y;
            }
            (Some(x), _) => {
                na = x;
            }
            (_, Some(y)) => {
                nb = y;
            }
            (_, _) => {}
        }
        let mut tmp_sum = na + nb + carry;
        carry = tmp_sum / 2;
        tmp_sum %= 2;
        res.push(tmp_sum);
    }
    if carry == 1 {
        res.push(1);
    }
    res.reverse();
    res.iter().map(|x| x.to_string()).collect()
}

pub fn reverse_bits(x: u32) -> u32 {
    let mut res = 0;
    for i in 0..32 {
        let t = 1 & (x >> i);
        res += t;
        if i < 31 {
            res = res << 1;
        }
    }
    res
}

pub fn hamming_weight(n: i32) -> i32 {
    let mut n = n;
    let mut res = 0;
    while n > 0 {
        n = n & (n-1);
        res += 1;
    }
    res
}

pub fn single_number(nums: Vec<i32>) -> i32 {
    let mut res = 0;
    for n in nums {
        res ^= n;
    }
    res
}

pub fn single_number_2(nums: Vec<i32>) -> i32 {
    let mut res = 0;
    for bit in 0..32 {
        let mut c = 0;
        for n in nums.iter() {
            c += (*n >> bit) & 1;
        }
        res += (c % 3) << bit;
    }
    res
}

pub fn range_bitwise_and(left: i32, right: i32) -> i32 {
    let mut right = right;
    while right > left {
        right &= (right - 1);
    }
    right
}

pub fn is_palindrome(x: i32) -> bool {
    if x < 0 {
        return false;
    }
    if x < 10 {
        return true;
    }
    if x % 10 == 0 {
        return false;
    }
    let (mut d, mut num) = (1, x);
    while num / d >= 10 {
        d *= 10;
    }
    while num != 0 {
        let (l, r) = (num/d, num%10);
        if l != r {
            return false;
        }
        num = (num - l * d) / 10;
        d /= 100;
    }
    true
}

pub fn trailing_zeroes(n: i32) -> i32 {
    let mut res = 0;
    let mut i = 5;
    while i <= n {
        let mut x = i;
        while i % 5 == 0 {
            res += 1;
            x /= 5;
        }
        i += 5;
    }
    res
}

pub fn my_sqrt(x: i32) -> i32 {
    let (mut l, mut r) = (0, x);
    let mut res = 0;
    while l <= r {
        let mid = l + ( r - l) / 2;
        if mid as i64 * mid as i64 > x as i64 {
            r = mid - 1;
        } else {
            res = mid;
            l = mid + 1;
        }
    }
    res
}

pub fn my_pow(x: f64, n: i32) -> f64 {
    if n == 0 {
        return  1 as f64;
    } else if n == 1 {
        return x;
    } else if n < 0 {
        return 1 as f64 / my_pow(x, -1*n);
    }

    let mut ret = 1 as f64;
    let mut n = n;
    let mut x = x;
    while n > 0 {
        if n & 1 == 1 {
            ret *= x;
        }
        x *= x;
        n /= 2;
    }
    ret
}


pub fn climb_stairs(n: i32) -> i32 {
    let mut dp = vec![0; n as usize+1];
    dp[0] = 1;
    dp[1] = 1;
    for i in 2..=n as usize {
        dp[i] = dp[i-1] + dp[i-2];
    }
    dp[n as usize]
}

pub fn rob(nums: Vec<i32>) -> i32 {
    let mut a = 0;
    let mut b = 0;
    for n in nums {
        (a, b) = (b, b.max(a+n));
    }
    b
}

pub fn word_break(s: String, word_dict: Vec<String>) -> bool {
    use std::collections::HashSet;

    let mut set = HashSet::new();
    for s in word_dict {
        set.insert(s);
    }
    let mut dp = vec![false; s.len()+1];
    dp[0] = true;
    for i in 1..=s.len() {
        for j in 0..i {
            if dp[j] && set.contains(&s[j..i]) {
                dp[i] = true;
                break;
            }
        }
    }
    dp[s.len()]
}

pub fn coin_change(coins: Vec<i32>, amount: i32) -> i32 {
    let mut dp = vec![i32::MAX; amount as usize + 1];
    dp[0] = 0;
    for coin in coins {
        let coin = coin as usize;
        for i in coin..amount as usize + 1 {
            if dp[i-coin] != i32::MAX {
                dp[i] = dp[i].min(dp[i-coin] + 1);
            }
        }
    }
    if dp[amount as usize] == i32::MAX {
        return -1;
    }
    dp[amount as usize]
}

pub fn length_of_lis(nums: Vec<i32>) -> i32 {
    if nums.len() == 0 {
        return 0;
    }
    let mut dp = vec![1; nums.len()];
    for i in 0..nums.len() {
        for j in 0..i {
            if nums[i] > nums[j] {
                dp[i] = dp[i].max(dp[j] + 1);
            }
        }
    }
    dp.into_iter().max().unwrap()
}

pub fn length_of_lis2(nums: Vec<i32>) -> i32 {
    let mut stack = vec![];
    for i in 0..nums.len() {
        if stack.len() == 0 || nums[i] > *stack.last().unwrap() {
            stack.push(nums[i]);
        } else {
            let index = match stack.binary_search(&nums[i]) {
                Ok(i) => i,
                Err(i) => i
            };
            stack[index] = nums[i];
        }
    }
    stack.len() as i32
}