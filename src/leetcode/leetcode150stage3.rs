
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