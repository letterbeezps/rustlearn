use std::mem::swap;



pub fn move_zeroes(nums: &mut Vec<i32>) {
    let mut pre = 0;
    for i in 0..nums.len() {
        if nums[i] != 0 {
            let tmp = nums[i];
            nums[i] = nums[pre];
            nums[pre] = tmp;
            pre += 1;
        }
    }
}