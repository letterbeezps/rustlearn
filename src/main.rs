use leetcode::leetcode::leetcode150;

fn main() {
    // let mut arr1 = vec![0,1,0,2,1,0,1,3,2,1,2,1];
    // let mut arr2 = vec!["This", "is", "an", "example", "of", "text", "justification."];
    // let arr3 :Vec<String> = arr2.into_iter().map(String::from).collect();
    // let s = String::from("A");
    let ret = leetcode150::merge_2(vec![vec![1,3], vec![2,6], vec![8,10],vec![15,18]]);
    // println!("{:?}", arr1);
    println!("{:?}", ret);
}
