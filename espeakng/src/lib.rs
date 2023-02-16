#![allow(non_upper_case_globals)]

use core::slice;
use std::{
    cell::Cell,
    ffi::{c_int, c_short, CStr, CString},
    time::Duration,
};

use espeakng_sys::*;

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
            name: std::ptr::null_mut(),
            languages: std::ptr::null(),
            identifier: ESPEAKNG_DEFAULT_VOICE.as_ptr().cast(),
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

#[derive(Clone)]
pub struct Fragment {
    pub data: Vec<i16>,
    pub duration: usize,
}

unsafe extern "C" fn callback(
    wave: *mut c_short,
    numsamples: c_int,
    events: *mut espeak_EVENT,
) -> c_int {
    let output: *mut Fragment = (*events).user_data.cast();
    let wave_slice = slice::from_raw_parts(wave, numsamples as usize);
    (*output).data.extend_from_slice(wave_slice);
    let events = EventIter { ptr: events };
    for event in events {
        if let espeak_EVENT_TYPE_espeakEVENT_END = event.type_ {
            (*output).duration = event.audio_position as usize
        }
    }
    0
}

impl EspeakNg {
    pub fn new() -> Result<Self, Error> {
        let error_context = std::ptr::null_mut();
        unsafe {
            espeak_ng_InitializePath(std::ptr::null());
            es_try!(espeak_ng_Initialize(error_context))?;
            es_try!(espeak_ng_InitializeOutput(0, 500, std::ptr::null()))?;
        }
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

    pub fn synthesize(&self, voice: Voice, utterance: impl AsRef<str>) -> Result<Fragment, Error> {
        let utt = [utterance.as_ref()];
        Ok(self.synthesize_multiple(voice, &utt)?.next().unwrap())
    }

    pub fn synthesize_multiple<S: AsRef<str>>(
        &self,
        mut voice: Voice,
        utterances: &[S],
    ) -> Result<impl Iterator<Item = Fragment>, Error> {
        let language = CString::new("en").unwrap();
        let mut voice_properties = espeak_VOICE {
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
            //es_try!(espeak_ng_SetVoiceByProperties((&mut voice.0) as *mut _))?;
            es_try!(espeak_ng_SetVoiceByProperties(
                &mut voice_properties as *mut _
            ))?;
            espeak_SetSynthCallback(Some(callback));
        }
        let mut output = vec![
            Fragment {
                data: Vec::new(),
                duration: 0
            };
            utterances.len()
        ];
        for (utterance, output) in utterances.iter().zip(output.iter_mut()) {
            let ptr = output as *mut Fragment;
            unsafe {
                let text = CString::new(utterance.as_ref()).unwrap();
                let text_len = text.as_bytes().len();
                // RETRIEVAL
                while es_try!(espeak_ng_Synthesize(
                    text.as_ptr() as *const _,
                    text_len as u64,
                    0,
                    0,
                    // We use a rust str, so we always have valid UTF-8
                    espeakCHARS_AUTO,
                    0,
                    std::ptr::null_mut(),
                    ptr as *mut _,
                ))
                .is_err()
                {
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
        }
        unsafe { espeak_ng_Synchronize() };
        Ok(output.into_iter())
    }
}
