#![allow(non_upper_case_globals)]

use core::slice;

use std::sync::{Arc, Condvar, Mutex, Once};
use std::{
    cell::Cell,
    ffi::{c_int, c_short, CStr, CString},
};

use espeakng_sys::*;

static ES_INIT: Once = Once::new();

macro_rules! es_try {
    ($e:expr) => {{
        let res = $e;
        match res {
            espeak_ng_STATUS_ENS_OK => Ok(()),
            _ => Err(Error { code: res }),
        }
    }};
}

pub struct EspeakNg {
    _error_context: *mut espeak_ng_ERROR_CONTEXT,
    sample_rate: Cell<Option<u32>>,
}

#[derive(Debug)]
pub struct Error {
    code: espeak_ng_STATUS,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const LEN: usize = 256;
        unsafe {
            let mut buf = [-1i8; LEN];
            let ptr = buf.as_mut_ptr();
            espeak_ng_GetStatusCodeMessage(self.code, buf.as_mut_ptr(), LEN as u64);
            let string = CStr::from_ptr(ptr);
            f.write_str(string.to_str().unwrap())
        }
    }
}

#[repr(i8)]
pub enum Gender {
    Male = 1,
    Female = 2,
    Neutral = 3,
}

#[repr(transparent)]
pub struct Voice(espeak_VOICE);

impl Default for Voice {
    fn default() -> Self {
        Self(espeak_VOICE {
            name: std::ptr::null(),
            languages: std::ptr::null(),
            identifier: std::ptr::null(),
            age: 0,
            gender: 0,
            variant: 0,
            xx1: 0,
            score: 0,
            spare: std::ptr::null_mut(),
        })
    }
}

macro_rules! replace_string {
    ($target:expr, $new:expr) => {
        let new_name = CString::new($new).unwrap();
        let old_name = $target;
        $target = new_name.into_raw();
        if !old_name.is_null() {
            unsafe { CString::from_raw(old_name as *mut i8) };
        }
    };
}

impl Voice {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn identifier(&mut self, identifier: &str) -> &mut Self {
        if self.0.identifier == ESPEAKNG_DEFAULT_VOICE.as_ptr().cast() {
            self.0.identifier = std::ptr::null();
        }
        replace_string!(self.0.identifier, identifier);
        self
    }

    pub fn name(&mut self, name: &str) -> &mut Self {
        replace_string!(self.0.name, name);
        self
    }

    pub fn age(&mut self, age: Option<u8>) -> &mut Self {
        self.0.age = age.unwrap_or(0);
        self
    }

    pub fn gender(&mut self, gender: Option<Gender>) -> &mut Self {
        self.0.gender = gender.map(|g| g as u8).unwrap_or(0);
        self
    }

    pub fn variant(&mut self, variant: u8) -> &mut Self {
        self.0.variant = variant;
        self
    }
}

impl std::error::Error for Error {}

struct EventIter {
    ptr: *mut espeak_EVENT,
}

impl Iterator for EventIter {
    type Item = espeak_EVENT;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            match (*self.ptr).type_ {
                espeak_EVENT_TYPE_espeakEVENT_LIST_TERMINATED => None,
                _ => {
                    let ret = *self.ptr;
                    self.ptr = self.ptr.add(1);
                    Some(ret)
                }
            }
        }
    }
}

#[derive(Clone, Default)]
pub struct Fragment {
    pub data: Vec<i16>,
    pub duration: usize,
}

unsafe extern "C" fn callback(
    wave: *mut c_short,
    numsamples: c_int,
    events: *mut espeak_EVENT,
) -> c_int {
    let output: *mut SynthData = (*events).user_data.cast();
    let wave_slice = slice::from_raw_parts(wave, numsamples as usize);
    (*output).fragment.data.extend_from_slice(wave_slice);
    let events = EventIter { ptr: events };
    for event in events {
        match event.type_ {
            espeak_EVENT_TYPE_espeakEVENT_END => {
                (*output).fragment.duration = event.audio_position as usize
            }
            espeak_EVENT_TYPE_espeakEVENT_MSG_TERMINATED => {
                let (c, m) = &*(*output).condvar;
                let mut g = m.lock().unwrap();
                *g = true;
                c.notify_one();
            }
            _ => (),
        }
    }
    0
}

pub struct SpokenIter<S: AsRef<str>, I: Iterator<Item = S>> {
    voice: Voice,
    utterances: I,
}

impl<S, I> SpokenIter<S, I>
where
    S: AsRef<str>,
    I: Iterator<Item = S>,
{
    fn new(voice: Voice, utterances: I) -> Self {
        Self { voice, utterances }
    }
}

struct SynthData {
    fragment: Fragment,
    condvar: Arc<(Condvar, Mutex<bool>)>,
}

impl<S, I> Iterator for SpokenIter<S, I>
where
    I: Iterator<Item = S>,
    S: AsRef<str>,
{
    type Item = Fragment;

    fn next(&mut self) -> Option<Self::Item> {
        let Some(utt) = self.utterances.next() else {
            unsafe {espeak_ng_Synchronize()};
            return None;
        };

        let condvar = Arc::new((Condvar::new(), Mutex::new(false)));
        let fragment = Fragment::default();
        let mut data = SynthData {
            condvar: condvar.clone(),
            fragment,
        };
        let language = CString::new("en").unwrap();
        let _voice_properties = espeak_VOICE {
            name: std::ptr::null(),
            languages: language.as_ptr(),
            identifier: std::ptr::null(),
            age: 0,
            gender: 0,
            variant: 0,
            xx1: 0,
            score: 0,
            spare: std::ptr::null_mut(),
        };
        unsafe {
            let text = CString::new(utt.as_ref()).unwrap();
            let text_len = text.as_bytes().len();
            // RETRIEVAL
            es_try!(espeak_ng_Synthesize(
                text.as_ptr() as *const _,
                text_len as u64,
                0,
                0,
                // We use a rust str, so we always have valid UTF-8
                espeakCHARS_AUTO,
                0,
                std::ptr::null_mut(),
                &mut data as *mut _ as *mut _,
            ))
            .unwrap();
        }
        let (c, m) = &*condvar;
        let _guard = c.wait_while(m.lock().unwrap(), |m: &mut bool| !*m).unwrap();
        Some(data.fragment)
    }
}

impl EspeakNg {
    pub fn new() -> Result<Self, Error> {
        let error_context = std::ptr::null_mut();
        ES_INIT.call_once(|| unsafe {
            espeak_Initialize(
                espeak_AUDIO_OUTPUT_AUDIO_OUTPUT_RETRIEVAL,
                500,
                std::ptr::null(),
                0,
            );
        });
        Ok(Self {
            _error_context: error_context,
            sample_rate: Cell::new(None),
        })
    }

    pub fn sample_rate(&self) -> u32 {
        let sample_rate = self.sample_rate.get();
        if let Some(sample_rate) = sample_rate {
            sample_rate
        } else {
            let sample_rate = unsafe { espeak_ng_GetSampleRate() as u32 };
            self.sample_rate.set(Some(sample_rate));
            sample_rate
        }
    }

    pub fn synthesize_multiple<S: AsRef<str>, I: Iterator<Item = S>>(
        &self,
        mut voice: Voice,
        utterances: I,
    ) -> Result<SpokenIter<S, I>, Error> {
        unsafe {
            es_try!(espeak_ng_SetVoiceByProperties(&mut voice.0 as *mut _))?;
            espeak_SetSynthCallback(Some(callback));
        }

        let iter = SpokenIter::new(voice, utterances);
        Ok(iter)
    }
}
