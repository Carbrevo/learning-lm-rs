#![allow(unused)]
use std::default;
use std::fmt::{Debug, Display, Error, Formatter};

use tokenizers::processors::template;
use tokenizers::Tokenizer;
use tokenizers::{tokenizer, Token};

use crate::kvcache::*;
use crate::logits::*;
use crate::model::*;
use crate::operators as OP;
use crate::tensor::*;

pub struct Generator<'a, T>
where
    T: Debug + Copy + Default,
{
    model: &'a Llama<T>,
    tokenizer: &'a Tokenizer,
    kvcache: KVCache<T>,
    logits_hist: Vec<Tensor<T>>,
    gen_path: Vec<u32>,
    txt_ids: Vec<u32>,
    pub(crate) inpos: usize,
    pub(crate) max_len: usize,
    pub(crate) top_p: f32,
    pub(crate) top_k: u32,
    pub(crate) temperature: f32,
}

impl<'a, T: Debug + Copy + Default> Generator<'a, T> {
    pub fn new(
        model: &'a Llama<T>,
        tokenizer: &'a Tokenizer,
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Self {
        Self {
            model,
            tokenizer,
            kvcache: KVCache::new(model.n_layers, max_len, model.n_kv_h * model.dqkv, 0),
            logits_hist: Vec::<Tensor<T>>::new(),
            gen_path: Vec::<u32>::new(),
            txt_ids: Vec::<u32>::new(),
            inpos: 0,
            max_len,
            top_p,
            top_k,
            temperature,
        }
    }

    pub fn prompt<'b: 'a>(&'b mut self, input: &str) -> GenerateIterator<'b, T> {
        let binding = self.tokenizer.encode(input, true).unwrap();
        let input_ids = binding.get_ids();
        let tokens = binding.get_tokens();
        //eprintln!("prompt: binding ids{:?}, tok{:?}", input_ids, tokens);
        //eprintln!("decode input_ids: {}", tokenizer.decode(&input_ids, true).unwrap());
        self.txt_ids = input_ids.to_vec();

        GenerateIterator::<'b, T>::new(self)
    }
}

pub struct GenerateIterator<'a, T>
where
    T: Debug + Copy + Default,
{
    ctxgen: &'a mut Generator<'a, T>,
    pos: usize,
}

impl<'a, T: Debug + Copy + Default> GenerateIterator<'a, T> {
    fn new(ctxgen: &'a mut Generator<'a, T>) -> Self {
        Self { ctxgen, pos: 0 }
    }
}

impl<'a> Iterator for GenerateIterator<'a, f32> {
    type Item = String;
    fn next(&mut self) -> Option<Self::Item> {
        let ctxgen = &mut self.ctxgen;
        let model = ctxgen.model;
        let tokenizer = ctxgen.tokenizer;

        let input = match ctxgen.txt_ids[ctxgen.inpos..] {
            ref i if i.len() == 0 => None,
            ref i if i[0] == model.eos_token_id => {
                ctxgen.inpos += 1;
                None
            }
            ref i => {
                ctxgen.inpos = ctxgen.txt_ids.len();
                Some(i)
            }
        };

        let logits: Option<Logits<f32>> = input.map(|i| {
            model
                .forward(
                    &Tensor::new(i.to_vec(), &vec![i.len()]),
                    &mut ctxgen.kvcache,
                )
                .into()
        });

        if let Some(token) = logits.map(|mut lgt| {
            lgt.sort();
            //print_logits(ctxgen.tokenizer, &lgt);
            lgt.random_sample(ctxgen.top_p, ctxgen.top_k, ctxgen.temperature)
            //lgt.argmax_sample()
            //lgt.beam_search()
        }) {
            ctxgen.txt_ids.push(token);
        }

        match ctxgen.txt_ids[self.pos..ctxgen.inpos] {
            ref ids if ids.len() == 0 => None,
            ref ids if ids[0] == model.eos_token_id => {
                self.pos += 1;
                Some("\n".to_string())
            }
            ref ids => {
                assert!(ids[0] != model.eos_token_id);
                assert!(ids.len() == 1 || self.pos == 0);
                self.pos = ctxgen.inpos;
                //print!("{}:",ids[0]);
                Some(tokenizer.decode(ids, true).unwrap())
            }
        }
    }
}

fn print_logits(tokenizer: &Tokenizer, logits: &Logits<f32>) {
    let first_chunk = logits.0.first_chunk::<16>().unwrap();
    first_chunk.iter().enumerate().for_each(|(idx, w)| {
        println!(
            "[{:2}]={:4}'{}':{}; ",
            idx,
            w.tok,
            tokenizer.decode(&[w.tok], true).unwrap(),
            w.val
        );
    })
}
