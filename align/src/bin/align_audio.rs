use align::{
    audioreader::AudioReader,
    fast_dtw::{dtw, dtw_sakoe_chuba, fast_dtw},
};
use mfcc::{mfcc, FrameExtractionOpts, MelBanksOpts, MfccOptions};

fn main() {
    let text = [
        "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL",
        "NOR IS MISTER QUILTER'S MANNER LESS INTERESTING THAN HIS MATTER",
    ];
    let a = "/tmp/append.flac";
    let b = "/tmp/middle.flac";

    let frame_opts = FrameExtractionOpts {
        sample_freq: 22050,
        frame_length_ms: 25.,
        frame_shift_ms: 10.,
        emphasis_factor: 0.97,
    };

    let mfcc_opts = MfccOptions {
        mel_opts: MelBanksOpts {
            n_bins: 40,
            low_freq: Some(133.3333.into()),
            high_freq: Some(6855.4976.into()),
        },
        frame_opts,
        n_ceps: 13,
    };

    let reader = AudioReader::new();
    let mut frames_a = reader.read_and_transcode_file(a, frame_opts).unwrap();
    let mut frames_b = reader.read_and_transcode_file(b, frame_opts).unwrap();

    let mfcc_a: Vec<_> = mfcc(mfcc_opts, &mut frames_a).collect();
    let mfcc_b: Vec<_> = mfcc(mfcc_opts, &mut frames_b).collect();

    let n = mfcc_a.len();
    let m = mfcc_b.len();
    let path = fast_dtw(&mfcc_a, &mfcc_b, Some(600), Some(1000));
    dbg!(path);
}
