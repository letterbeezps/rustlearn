use serde_derive::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct ParamA {
    name: String,
    age: u8,
}

#[derive(Serialize, Deserialize)]
struct ParamAA {
    name: String,
    age: u8,
    gender: bool,
}

#[derive(Serialize, Deserialize)]
struct ParamB {
    weight: u8,
    height: u8,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum SpecialParam  {
    AA(ParamAA),
    A(ParamA),
    B(ParamB),
}

#[cfg(test)]
mod test {
    use serde_json::value;

    use super::*;

    #[test]
    fn test_enum() {
        let a = SpecialParam::A(ParamA {
            name: String::from("a"),
            age: 1,
        });
        let aa = SpecialParam::AA(ParamAA {
            name: String::from("aa"),
            age: 1,
            gender: true,
        });
        let b = SpecialParam::B(ParamB {
            weight: 1,
            height: 1,
        });

        let a_json = serde_json::to_string(&a).unwrap();
        let aa_json = serde_json::to_string(&aa).unwrap();
        let b_json = serde_json::to_string(&b).unwrap();

        assert_eq!(a_json, r#"{"name":"a","age":1}"#);
        assert_eq!(aa_json, r#"{"name":"aa","age":1,"gender":true}"#);
        assert_eq!(b_json, r#"{"weight":1,"height":1}"#);

        let d_a: SpecialParam = serde_json::from_str(&a_json).unwrap();
        let d_aa: SpecialParam = serde_json::from_str(&aa_json).unwrap();
        let d_b: SpecialParam = serde_json::from_str(&b_json).unwrap();

        assert!(matches!(d_a, SpecialParam::A(_)));
        assert!(matches!(d_aa, SpecialParam::AA(_)));
        assert!(matches!(d_b, SpecialParam::B(_)));
    }
}