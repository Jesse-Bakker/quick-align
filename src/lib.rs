use core::slice;
use ndarray::linalg::general_mat_vec_mul;
use ndarray::prelude::*;
use realfft::num_complex::Complex;
use realfft::RealFftPlanner;

pub trait Feature {
    fn frame_options(&self) -> &FrameExtractionOpts;
    fn compute(&mut self, signal_frame: &mut [Float], feature: &mut [Float]);
    fn n_coeffs(&self) -> usize;
}

type Float = f32;
const PI: Float = std::f32::consts::PI;
const TWOPI: Float = std::f32::consts::TAU;

mod freq;

fn dot(v0: &[Float], v1: &[Float]) -> Float {
    v0.iter().zip(v1.iter()).map(|(e0, e1)| e0 * e1).sum()
}
struct MelBanks {
    center_freqs: Vec<freq::Freq>,
    bins: Vec<(usize, Array1<Float>)>,
}

struct MelBanksOpts {
    n_bins: usize,
    low_freq: freq::Freq,
    high_freq: freq::Freq,
}

#[derive(Clone)]
pub struct FrameExtractionOpts {
    sample_freq: freq::Freq,
    frame_length_ms: Float,
    frame_shift_ms: Float,
    emphasis_factor: Float,
}

impl FrameExtractionOpts {
    fn win_size(&self) -> usize {
        let freq: Float = (self.sample_freq * 0.001 * self.frame_length_ms).into();
        freq as usize
    }

    fn win_size_padded(&self) -> usize {
        self.win_size().next_power_of_two()
    }

    fn win_shift(&self) -> usize {
        Float::from(self.sample_freq * 0.001 * self.frame_shift_ms) as usize
    }
}

impl MelBanks {
    fn new(opts: &MelBanksOpts, frame_opts: &FrameExtractionOpts) -> Self {
        let MelBanksOpts {
            n_bins,
            low_freq,
            high_freq,
        } = opts;
        let n_bins = *n_bins as usize;
        let sample_freq = frame_opts.sample_freq;

        let win_size = frame_opts.win_size().next_power_of_two();
        let n_fft_bins = win_size / 2;

        let fft_bin_width = sample_freq / win_size as Float;

        let mel_low_freq: freq::MelFreq = low_freq.to_mel();
        let mel_high_freq: freq::MelFreq = high_freq.to_mel();

        let mel_freq_delta = (mel_high_freq - mel_low_freq) / (n_bins + 1) as Float;

        let (center_freqs, bins): (Vec<_>, Vec<_>) = (0..n_bins)
            .map(|bin| {
                let bin = bin as Float;
                let left_mel = mel_low_freq + bin * mel_freq_delta;
                let center_mel = mel_low_freq + (bin + 1.0) * mel_freq_delta;
                let right_mel = mel_low_freq + (bin + 2.0) * mel_freq_delta;

                let center_freq = center_mel.to_freq();
                let mut first_index = None;

                let this_bin: Array1<_> = (0..n_fft_bins)
                    .filter_map(|i| {
                        let freq: freq::Freq = fft_bin_width * i as Float;
                        let mel: freq::MelFreq = freq.to_mel();

                        if mel < left_mel || mel > right_mel {
                            return None;
                        }

                        let weight = if mel <= center_mel {
                            (mel - left_mel) / (center_mel - left_mel).into()
                        } else {
                            (right_mel - mel) / (right_mel - center_mel).into()
                        };
                        first_index.get_or_insert(i);
                        Some(weight.into())
                    })
                    .collect();
                (center_freq, (first_index.unwrap(), this_bin))
            })
            .unzip();
        Self { bins, center_freqs }
    }

    fn apply_in(&self, power_spectrum: ArrayView1<Float>, mel_energies: ArrayViewMut1<Float>) {
        for (bin, output) in self.bins.iter().zip(mel_energies.into_iter()) {
            let offset = bin.0;
            let bin_data = &bin.1;

            let energy = bin_data.dot(&power_spectrum.slice(s![offset..bin_data.len()]));
            *output = energy
        }
    }
}

pub struct Mfcc {
    mel_banks: MelBanks,
    dct_matrix: Array2<Float>,
    options: MfccOptions,
    mel_energies: Array1<Float>, // Temporary workspace
    fft_planner: RealFftPlanner<Float>,
    fft_scratch: Vec<Complex<Float>>,
    fft_out: Vec<Complex<Float>>,
}

impl Mfcc {
    fn dct_matrix(rows: usize, cols: usize) -> Array2<Float> {
        let mut matrix = Array2::zeros((rows, cols));

        let normalizer = (1.0 / cols as Float).sqrt();
        for col in 0..cols {
            matrix[(0, col)] = normalizer;
        }

        let normalizer = (2.0 / cols as f64).sqrt();
        for row in 1..rows {
            for col in 0..cols {
                matrix[(row, col)] = (normalizer
                    * f64::cos(PI as f64 / cols as f64 * (col as f64 + 0.5) * row as f64))
                    as Float
            }
        }
        matrix
    }

    pub fn new(options: MfccOptions) -> Self {
        let n_bins = options.mel_opts.n_bins;
        let dct_matrix = Self::dct_matrix(options.n_ceps, n_bins);
        let mel_banks = MelBanks::new(&options.mel_opts, &options.frame_opts);
        let win_size_padded = options.frame_opts.win_size_padded();
        let mut planner = RealFftPlanner::new();
        let fft = planner.plan_fft_forward(win_size_padded);
        let fft_scratch = fft.make_scratch_vec();
        let fft_out = fft.make_output_vec();
        Self {
            mel_banks,
            dct_matrix,
            options,
            mel_energies: Array1::zeros(n_bins),
            fft_planner: planner,
            fft_scratch,
            fft_out,
        }
    }

    fn compute_power_spectrum(spectrum: &mut [Complex<Float>]) -> &mut [Float] {
        let dim = spectrum.len();
        let ptr = spectrum.as_mut_ptr();
        unsafe {
            // We do some magic here to reuse the first half of the spectrum allocation
            // for our power spectrum
            let output = slice::from_raw_parts_mut(ptr.cast(), dim * 2);
            for i in 0..dim {
                let re = output[2 * i];
                let im = output[2 * i + 1];
                output[i] = re * re + im * im;
            }
            &mut output[0..dim]
        }
    }
}

impl Feature for Mfcc {
    fn compute(&mut self, frame: &mut [Float], feature: &mut [Float]) {
        // TODO: use raw spectral power, before windowing and pre-emphasis, as C0
        let mel_banks = &self.mel_banks;

        let fft = self
            .fft_planner
            .plan_fft_forward(self.options.frame_opts.win_size_padded());
        fft.process_with_scratch(
            frame,
            self.fft_out.as_mut_slice(),
            self.fft_scratch.as_mut_slice(),
        )
        .unwrap();

        let spectrum = &mut self.fft_out;
        let power_spectrum = ArrayViewMut::from(Mfcc::compute_power_spectrum(spectrum));

        mel_banks.apply_in(power_spectrum.view(), self.mel_energies.view_mut());
        for energy in self.mel_energies.iter_mut() {
            *energy = energy.ln();
        }

        general_mat_vec_mul(
            1.0,
            &self.dct_matrix,
            &self.mel_energies,
            0.0,
            &mut ArrayViewMut::from(feature),
        )

        // TODO: Cepstral liftering
    }

    fn frame_options(&self) -> &FrameExtractionOpts {
        &self.options.frame_opts
    }

    fn n_coeffs(&self) -> usize {
        self.options.n_ceps
    }
}

pub struct MfccOptions {
    mel_opts: MelBanksOpts,
    frame_opts: FrameExtractionOpts,
    n_ceps: usize,
}

pub struct OfflineFeature<T: Feature> {
    computer: T,
}

fn hamming_window(len: usize) -> Array1<Float> {
    let a = TWOPI / (len - 1) as Float;
    Array1::from_iter((0..len).map(|i| 0.54 - 0.46 * f32::cos(a * i as f32)))
}

impl<T: Feature> OfflineFeature<T> {
    fn preemphasize(frame: &mut [Float], opts: &FrameExtractionOpts) {
        let emph_fact = opts.emphasis_factor;
        for j in 1..frame.len() {
            frame[j] -= emph_fact * frame[j - 1];
        }
        frame[0] -= emph_fact * frame[0];
    }

    pub fn compute(&mut self, wave: &[Float]) -> Array2<Float> {
        let n_samples = wave.len();
        let opts = self.computer.frame_options().clone();
        let frame_shift = opts.win_shift();
        let frame_length_padded = self.computer.frame_options().win_size_padded();
        let num_frames = n_samples / frame_shift;

        let hamming_window = hamming_window(frame_length_padded);
        let mut frame = Array1::zeros(frame_length_padded);
        let mut output = Array2::zeros((num_frames, self.computer.n_coeffs()));
        for i in 0..num_frames {
            let frame_start = i * frame_shift;
            let frame_end = usize::min(frame_start + frame_length_padded, n_samples);
            let frame_view = ArrayView1::from(&wave[frame_start..frame_end]);
            frame
                .slice_mut(s![0..frame_end - frame_start])
                .assign(&frame_view);
            // Zero-pad the last frame
            if frame_start + frame_length_padded > n_samples {
                frame.slice_mut(s![frame_end - frame_start..]).fill(0.);
            }

            Self::preemphasize(frame.as_slice_mut().unwrap(), &opts);

            frame *= &hamming_window;
            let mut feature = output.slice_mut(s![i, ..]);

            self.computer.compute(
                frame.as_slice_mut().unwrap(),
                feature.as_slice_mut().unwrap(),
            )
        }
        output
    }
}
