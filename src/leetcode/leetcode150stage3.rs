


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