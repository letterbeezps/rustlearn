use std::{cell::RefCell, rc::Rc};


#[derive(Debug)]
enum List {
    Cons(Rc<RefCell<i32>>, Rc<List>),
    Nil,
}

use List::{Cons, Nil};

fn data_rc() {
    let v = RefCell::new(5);
    println!("data before {:?}", v.borrow());

    let mut v_borrow_mut = v.borrow_mut();
    *v_borrow_mut += 2;
    
    // drop(v_borrow_mut);

    println!("data after {:?}", v.borrow());
}

fn list_rc_refcell() {
    let value = Rc::new(RefCell::new(5));

    let a = Rc::new(Cons(Rc::clone(&value), Rc::new(Nil)));

    let b = Cons(Rc::new(RefCell::new(3)), Rc::clone(&a));
    let c = Cons(Rc::new(RefCell::new(4)), Rc::clone(&a));

    *value.borrow_mut() += 10;

    println!("a after = {:?}", a);
    println!("b after = {:?}", b);
    println!("c after = {:?}", c);
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_data_rc() {
        data_rc()
    }

    #[test]
    fn test_list_rc_refcell() {
        list_rc_refcell()
    }
}