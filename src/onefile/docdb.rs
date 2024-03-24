use core::fmt;
use std::{collections::{hash_map, HashMap}, fmt::Display, fs, io, path::{Path, PathBuf}, result, str::Utf8Error};

use serde::{de::DeserializeOwned, Serialize};

struct DocSerializer {}

impl DocSerializer {
    fn serialize_data<V>(&self, data: &V) -> Result<Vec<u8>>
    where V: Serialize,
    {
        let value = serde_json::to_string(data)?;
        Ok(value.into_bytes())
    }

    fn deserialize_data<V>(&self, data: &[u8]) -> Result<V>
    where V: DeserializeOwned
    {
        let value = serde_json::from_str(std::str::from_utf8(data)?)?;
        Ok(value)
    }

    fn serialize_db(&self, map: &HashMap<String, Vec<u8>>) -> Result<Vec<u8>>
    {
        let mut db_map: HashMap<_, _> = HashMap::new();
        db_map = map.iter().map(|(k, v)| (k, std::str::from_utf8(v).unwrap())).collect();
        let ret = serde_json::to_string(&db_map)?;
        Ok(ret.into_bytes())
    }
    
    fn deserialize_db(&self, db: &[u8]) -> Result<HashMap<String, Vec<u8>>> {
        let db_data = std::str::from_utf8(db)?;
        let data = serde_json::from_str::<HashMap<String, String>>(db_data)?;
        let mut ret = HashMap::new();
        ret = data.iter().map(|(k,v)| (k.to_string(), v.as_bytes().to_vec())).collect();
        Ok(ret)
    }
}

fn load(path: impl AsRef<Path>) -> Result<DocDB> {
    let content = fs::read(path.as_ref())?;
    let map = DocSerializer{}.deserialize_db(&content)?;
    Ok(DocDB{
        map,
        path: path.as_ref().to_path_buf(),
        serilizer: DocSerializer{},
    })
}


struct DocDB {
    map: HashMap<String, Vec<u8>>,
    path: PathBuf,
    serilizer: DocSerializer,
}

impl DocDB {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            map: HashMap::new(),
            path: path.as_ref().to_path_buf(),
            serilizer: DocSerializer{},
        }
    }

    pub fn set<V>(&mut self, key: &str, value: &V) -> Result<()>
    where
        V: Serialize,
    {
        let data = self.serilizer.serialize_data(value)?;
        let origin_data = self.map.insert(String::from(key), data);
        match self.dump() {
            Ok(_) => Ok(()),
            Err(err) => {
                match origin_data {
                    None => {
                        self.map.remove(key);
                    }
                    Some(origin_data) => {
                        self.map.insert(String::from(key), origin_data);
                    }
                }

                Err(err)
            }
        }
        
    }

    pub fn get<V>(&self, key: &str) -> Option<V>
    where
        V: DeserializeOwned,
    {
        let data = self.map.get(key)?;
        let val = self.serilizer.deserialize_data(data).unwrap();
        Some(val)
    }

    pub fn dump(&mut self) -> Result<()> {
        let db = self.serilizer.serialize_db(&self.map)?;
        let tmp_path = format!("{}.tmp", self.path.to_str().unwrap());
        fs::write(&tmp_path, db)?;

        fs::rename(&tmp_path, &self.path)?;
        Ok(())
    }
}

//////////////////////////  error type ////////////////////////

#[derive(Debug)]
enum DocError {
    IO(io::Error),
    Serialization(String),
    DeSerialization(String),
}

impl Display for DocError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DocError::IO(err) => fmt::Display::fmt(err, f),
            DocError::Serialization(v) => f.write_str(&format!("Serialization err: {}", v)),
            DocError::DeSerialization(v) => f.write_str(&format!("DeSerialization err: {}", v)),
        }
    }
}

type Result<T> = result::Result<T, DocError>;

impl From<serde_json::Error> for DocError {
    fn from(value: serde_json::Error) -> Self {
        DocError::Serialization(value.to_string())
    }
}

impl From<Utf8Error> for DocError {
    fn from(value: Utf8Error) -> Self {
        DocError::DeSerialization(value.to_string())
    }
}

impl From<io::Error> for DocError {
    fn from(value: io::Error) -> Self {
        DocError::IO(value)
    }
}

#[cfg(test)]
mod test {
    use super::{load, DocDB};

    
    #[test]
    fn test_get_set() {
        let mut db = DocDB::new("test.db");
        assert_eq!(None, db.get::<String>("key"));

        db.set("key", &String::from("value")).unwrap();
        assert_eq!(String::from("value"), db.get::<String>("key").unwrap())
    }

    #[test]
    fn test_serialize_db() {
        let mut db = DocDB::new("test.db");
        db.set("key", &String::from("value")).unwrap();
        let ret = db.serilizer.serialize_db(&db.map).unwrap();
        println!("{:?}", &ret);

        let db_map = db.serilizer.deserialize_db(&ret).unwrap();
        println!("{:?}", db_map);
        db.map = db_map;
        assert_eq!(String::from("value"), db.get::<String>("key").unwrap())
    }

    #[test]
    fn test_dump_load() {
        let mut db = DocDB::new("test.db");
        db.set("key", &String::from("value")).unwrap();
        db.set("key1", &vec![1,2,3]).unwrap();
        
        let new_db = load("test.db").unwrap();
        assert_eq!(String::from("value"), new_db.get::<String>("key").unwrap());
        assert_eq!(vec![1,2,3], new_db.get::<Vec<i32>>("key1").unwrap());
    }
}