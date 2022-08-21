#![allow(non_upper_case_globals)]

use espeakng_sys::*;
use lazy_static::lazy_static;
use std::ffi::{c_void, CString};
use std::os::raw::{c_char, c_int, c_short};
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

/// The length in mS of sound buffers passed to the SynthCallback function.
const BUFF_LEN: i32 = 500;
/// Options to set for espeak-ng
const OPTIONS: i32 = 0;

#[derive(Default)]
struct State {
    buffer: Vec<i16>,
    anchors: Vec<usize>,
    last_event_time: i32,
}
lazy_static! {
    /// The complete audio provided by the callback
    static ref STATE: Mutex<State> = Mutex::default();
    static ref GLOBAL_LOCK: Mutex<()> = Mutex::default();
}

/// Spoken speech
#[derive(Debug)]
pub struct Spoken {
    /// The audio data
    pub wav: Vec<i16>,
    /// The sample rate of the audio
    pub sample_rate: i32,
    pub anchors: Vec<usize>,
}

macro_rules! espeak_try {
    ($try:expr) => {
        match $try {
            espeak_ERROR_EE_OK => Ok(()),
            _ => Err(()),
        }
    };
}

/// Perform Text-To-Speech
pub(crate) fn speak_multiple(utterances: Vec<&str>) -> Result<Spoken, ()> {
    // Keep the global lock for the entire duration of synthesis.
    let _guard = GLOBAL_LOCK.plock();
    let output: espeak_AUDIO_OUTPUT = espeak_AUDIO_OUTPUT_AUDIO_OUTPUT_RETRIEVAL;

    {
        let mut state = STATE.plock();
        *state = Default::default();
    }
    // The directory which contains the espeak-ng-data directory, or NULL for the default location.
    let path: *const c_char = std::ptr::null();

    // Returns: sample rate in Hz, or -1 (EE_INTERNAL_ERROR).
    let sample_rate = unsafe { espeak_Initialize(output, BUFF_LEN, path, OPTIONS) };
    if sample_rate == espeak_ERROR_EE_INTERNAL_ERROR {
        return Err(());
    }

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
        espeak_try!(espeak_SetVoiceByProperties(&mut voice_properties as *mut _)).unwrap();
        espeak_SetSynthCallback(Some(synth_callback))
    }

    let position = 0u32;
    let position_type: espeak_POSITION_TYPE = 0;
    let end_position = 0u32;
    let flags = espeakCHARS_AUTO;
    let identifier = std::ptr::null_mut();
    let user_data = std::ptr::null_mut();

    let mut utterances = utterances.into_iter();
    let mut utterance = utterances.next();
    while let Some(current_utterance) = utterance {
        let text_cstr = CString::new(current_utterance).expect("Failed to convert &str to CString");
        unsafe {
            if (espeak_try!(espeak_Synth(
                text_cstr.as_ptr() as *const c_void,
                BUFF_LEN as size_t,
                position,
                position_type,
                end_position,
                flags,
                identifier,
                user_data,
            )))
            .is_ok()
            {
                utterance = utterances.next();
            } else {
                std::thread::sleep(Duration::from_millis(500));
            }
        }
    }

    // Wait for the speaking to complete
    unsafe { espeak_try!(espeak_Synchronize()).unwrap() };

    let state_guard = STATE.plock();
    unsafe { espeak_try!(espeak_Terminate()).unwrap() };

    Ok(Spoken {
        wav: state_guard.buffer.clone(),
        sample_rate,
        anchors: state_guard.anchors.clone(),
    })
}

/// int SynthCallback(short *wav, int numsamples, espeak_EVENT *events);
///
/// wav:  is the speech sound data which has been produced.
/// NULL indicates that the synthesis has been completed.
///
/// numsamples: is the number of entries in wav.  This number may vary, may be less than
/// the value implied by the buflength parameter given in espeak_Initialize, and may
/// sometimes be zero (which does NOT indicate end of synthesis).
///
/// events: an array of espeak_EVENT items which indicate word and sentence events, and
/// also the occurance if <mark> and <audio> elements within the text.  The list of
/// events is terminated by an event of type = 0.
///
/// Callback returns: 0=continue synthesis,  1=abort synthesis.
unsafe extern "C" fn synth_callback(
    wav: *mut c_short,
    sample_count: c_int,
    events: *mut espeak_EVENT,
) -> c_int {
    let mut state = STATE.plock();
    // Calculate the length of the events array
    let mut events = events;

    // Turn the audio wav data array into a Vec.
    // We must clone from the slice, as the provided array's memory is managed by C
    let wav_slice = std::slice::from_raw_parts_mut(wav, sample_count as usize);
    let mut wav_vec = wav_slice
        .iter_mut()
        .map(|f| *f as i16)
        .collect::<Vec<i16>>();

    while (*events).type_ != espeak_EVENT_TYPE_espeakEVENT_LIST_TERMINATED {
        // Determine if this is the end of the synth
        let event = *events;
        match event.type_ {
            espeak_EVENT_TYPE_espeakEVENT_END => state.last_event_time = event.audio_position,

            espeak_EVENT_TYPE_espeakEVENT_MSG_TERMINATED => {
                let time = state.anchors.last().unwrap_or(&0) + state.last_event_time as usize;
                state.anchors.push(time);
            }
            _ => {}
        }
        events = events.add(1);
    }

    state.buffer.append(&mut wav_vec);

    0
}

trait PoisonlessLock<T> {
    fn plock(&self) -> MutexGuard<T>;
}

impl<T> PoisonlessLock<T> for Mutex<T> {
    fn plock(&self) -> MutexGuard<T> {
        match self.lock() {
            Ok(l) => l,
            Err(e) => e.into_inner(),
        }
    }
}
