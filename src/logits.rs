#![allow(unused)]
use crate::tensor::*;
use std::cmp::*;
use std::fmt::{Debug, Error, Formatter};

#[derive(Clone, Copy, Debug)]
pub struct Weight<T: PartialOrd + PartialEq> {
    pub val: T,
    pub tok: u32,
}
impl<T: PartialOrd + PartialEq> PartialEq for Weight<T> {
    fn eq(&self, other: &Self) -> bool {
        self.val.eq(&other.val)
    }
}

impl<T: PartialOrd + PartialEq> PartialOrd for Weight<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.val.partial_cmp(&other.val).unwrap().reverse())
    }
}

impl<T: PartialEq + PartialOrd + Copy + Debug> From<(usize, &T)> for Weight<T> {
    #[inline]
    fn from((t, v): (usize, &T)) -> Self {
        Self {
            val: *v,
            tok: t as _,
        }
    }
}

pub struct Logits<T: PartialEq + PartialOrd>(pub(crate) Vec<Weight<T>>);

impl<T: PartialEq + PartialOrd + Copy + Default + Debug> Logits<T> {
    pub fn new(x: &Tensor<T>) -> Self {
        assert!(x.shape()[x.shape().len() - 1] == x.size());
        Self(
            x.data()
                .iter()
                .enumerate()
                .map(Weight::<T>::from)
                .collect::<Vec<_>>(),
        )
    }

    pub fn sort(&mut self) {
        self.0.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    }

    #[allow(unused)]
    pub fn argmax_sample(&mut self) -> u32 {
        self.0[0].tok
    }

    #[allow(unused)]
    pub fn beam_search(&mut self) -> u32 {
        2
    }
}

impl Logits<f32> {
    #[allow(unused)]
    pub fn random_sample(&mut self, top_p: f32, top_k: u32, temperature: f32) -> u32 {
        let max = core::mem::replace(&mut self.0[0].val, 1.);
        // softmax & sum
        for i in 1..self.0.len() {
            self.0[i].val = self.0[i - 1].val + ((self.0[i].val - max) / temperature).exp();
        }
        // topk & topp & random
        let pk = self.0[(top_k as usize).min(self.0.len()) - 1].val;
        let pp = self.0[self.0.len() - 1].val * top_p;
        let plimit = rand::random::<f32>() * f32::min(pk, pp);
        // sample
        self.0.iter().find(|p| p.val >= plimit).unwrap().tok
    }
}

impl<T: PartialEq + PartialOrd + Copy + Default + Debug> From<Tensor<T>> for Logits<T> {
    fn from(tensor: Tensor<T>) -> Logits<T> {
        Logits::<T>::new(&tensor)
    }
}

impl<T: PartialEq + PartialOrd + Copy + Default + Debug> Debug for Logits<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        let first_chunk = self.0.first_chunk::<16>().unwrap();
        let mut result = Err(Error::default());
        for w in first_chunk {
            result = write!(f, "[{}]: '{}' = {:?}\n", w.tok, w.tok, w.val);
        }
        result
    }
}
