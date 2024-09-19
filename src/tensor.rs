#![allow(dead_code)]
#![allow(unused)]

use std::{slice, sync::Arc, vec};
use std::fmt::{ Display, Formatter, Debug, Error };
use std::ops::{ Index };

#[derive(Clone, Default)]
pub struct Tensor<T> 
where T: Debug {
    data: Arc<Box<[T]>>,
    shape: Vec<usize>,
    offset: usize,
    length: usize,
}

impl<T : Debug > Display for Tensor<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        if let Some(first) = self.data.first_chunk::<16>() {
            write!(f, "Tensor: Shape={:?}, off={}, len={}\nData: {:?}", 
                    self.shape, self.offset, self.length, first)
        } else {
            write!(f, "Tensor: Shape={:?}, off={}, len={}\nData: {:?}", 
                    self.shape, self.offset, self.length, self.data)
        }
    }
}

//impl<T: Debug> Index<usize> for Tensor<T> {

//}

impl<T: Copy + Clone + Default + Debug> Tensor<T> {
    pub fn new(data: Vec<T>, shape: &Vec<usize>) -> Self {
        let length = data.len();
        Tensor {
            data: Arc::new(data.into_boxed_slice().try_into().unwrap()),
            shape: shape.clone(),
            offset: 0,
            length: length,
        }
    }

    pub fn default(shape: &Vec<usize>) -> Self {
        let length = shape.iter().product();
        let data = vec![T::default(); length];
        Self::new(data, shape)
    }

    pub fn data(&self) -> &[T] {
        &self.data[self.offset..][..self.length]
    }

    pub unsafe fn data_mut(&mut self) -> &mut [T] {
        let ptr = self.data.as_ptr().add(self.offset) as *mut T;
        slice::from_raw_parts_mut(ptr, self.length)
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn size(&self) -> usize {
        self.length
    }

    // Reinterpret the tensor as a new shape while preserving total size.
    pub fn reshape(&mut self, new_shape: &Vec<usize>) -> &mut Self {
        let new_length: usize = new_shape.iter().product();
        if new_length != self.length {
            let old_shape = self.shape.clone();
            panic!("New shape {new_shape:?} does not match tensor of {old_shape:?}");
        }
        self.shape = new_shape.clone();
        self
    }

    pub fn slice(&self, start: usize, shape: &Vec<usize>) -> Self {
        let new_length: usize = shape.iter().product();
        assert!(self.offset + start + new_length <= self.length);
        Tensor {
            data: self.data.clone(),
            shape: shape.clone(),
            offset: self.offset + start,
            length: new_length,
        }
    }
}

// Some helper functions for testing and debugging
impl Tensor<f32> {
    #[allow(unused)]
    pub fn close_to(&self, other: &Self, rel: f32) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        let a = self.data();
        let b = other.data();
        
        return a.iter().zip(b).all(|(x, y)| float_eq(x, y, rel));
    }
    #[allow(unused)]
    pub fn print(&self){
        println!("shpae: {:?}, offset: {}, length: {}", self.shape, self.offset, self.length);
        let dim = self.shape()[self.shape().len() - 1];
        let batch = self.length / dim;
        for i in 0..batch {
            let start = i * dim;
            println!("{:?}", &self.data()[start..][..dim]);
        }
    }
}

#[inline]
pub fn float_eq(x: &f32, y: &f32, rel: f32) -> bool {
    if (x - y).abs() <= rel * (x.abs() + y.abs()) / 2.0 {
        println!("f32({}=={}) rel={} byte: {:?} == {:?}", x,y, rel * (x.abs() + y.abs()) / 2.0,
                                                        x.to_le_bytes(), y.to_le_bytes());
        true
    } else {
        println!("f32({}!={}) rel={} byte: {:?} != {:?}", x,y, rel * (x.abs() + y.abs()) / 2.0,
                                                x.to_le_bytes(), y.to_le_bytes());
        false
    }
}
