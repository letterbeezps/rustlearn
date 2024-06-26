
enum List {
    Cons(i32, Rc<List>),
    Nil,
}

use std::rc::Rc;

use List::{Cons, Nil};
fn list_with_box() {
    let a = Rc::new(Cons(5, Rc::new(Cons(10, Rc::new(Nil)))));
    println!("count after creating a = {}", Rc::strong_count(&a));
    let b = Cons(3, Rc::clone(&a));
    println!("count after creating b = {}", Rc::strong_count(&a));
    {
        let c = Cons(4, Rc::clone(&a));
        println!("count after creating c = {}", Rc::strong_count(&a));
    }
    println!("count after c goes out of scope = {}", Rc::strong_count(&a));
}

#[cfg(test)]
mod test {
    use super::list_with_box;


    #[test]
    fn test_list_with_box() {
        list_with_box()
    }
}