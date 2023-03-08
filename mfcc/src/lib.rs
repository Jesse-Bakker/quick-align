use core::slice;
use freq::Freq;
use ndarray::linalg::general_mat_vec_mul;
use ndarray::prelude::*;
use realfft::num_complex::Complex;
use realfft::RealFftPlanner;

pub trait Feature {
    fn frame_options(&self) -> &FrameExtractionOpts;
    fn compute(&mut self, signal_frame: ArrayViewMut1<Float>, feature: ArrayViewMut1<Float>);
    fn n_coeffs(&self) -> usize;
}

type Float = f32;
const PI: Float = std::f32::consts::PI;
const TWOPI: Float = std::f32::consts::TAU;

pub mod freq;

struct MelBanks {
    _center_freqs: Vec<freq::Freq>,
    bins: Vec<(usize, Array1<Float>)>,
}

pub struct MelBanksOpts {
    pub n_bins: usize,
    pub low_freq: Option<freq::Freq>,
    pub high_freq: Option<freq::Freq>,
}

#[derive(Clone, Copy)]
pub struct FrameExtractionOpts {
    pub sample_freq: u32,
    pub frame_length_ms: Float,
    pub frame_shift_ms: Float,
    pub emphasis_factor: Float,
}

impl FrameExtractionOpts {
    pub fn win_size(&self) -> usize {
        let freq: Float = self.sample_freq as Float * 0.001 * self.frame_length_ms;
        freq as usize
    }

    pub fn win_size_padded(&self) -> usize {
        self.win_size().next_power_of_two()
    }

    pub fn win_shift(&self) -> usize {
        (self.sample_freq as Float * 0.001 * self.frame_shift_ms) as usize
    }
}

impl MelBanks {
    fn new(opts: &MelBanksOpts, frame_opts: &FrameExtractionOpts) -> Self {
        let MelBanksOpts {
            n_bins,
            low_freq,
            high_freq,
        } = opts;
        let n_bins = *n_bins;
        let sample_freq = Freq::from(frame_opts.sample_freq as f32);
        let nyquist = 0.5 * sample_freq;

        let win_size = frame_opts.win_size().next_power_of_two();
        let n_fft_bins = win_size / 2;

        let fft_bin_width = sample_freq / win_size as Float;

        let mel_low_freq: freq::MelFreq = low_freq.unwrap_or(0.0.into()).to_mel();
        let mel_high_freq: freq::MelFreq = high_freq.unwrap_or(nyquist).to_mel();

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
                        let freq: freq::Freq = fft_bin_width * (i as Float);
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
        Self {
            bins,
            _center_freqs: center_freqs,
        }
    }

    fn apply_in(&self, power_spectrum: ArrayView1<Float>, mel_energies: ArrayViewMut1<Float>) {
        for (bin, output) in self.bins.iter().zip(mel_energies.into_iter()) {
            let offset = bin.0;
            let bin_data = &bin.1;

            let energy = bin_data.dot(&power_spectrum.slice(s![offset..offset + bin_data.len()]));
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
    fn compute(&mut self, mut frame: ArrayViewMut1<Float>, mut feature: ArrayViewMut1<Float>) {
        // XXX: use raw spectral power, before windowing and pre-emphasis, as C0
        let mel_banks = &self.mel_banks;

        let fft = self
            .fft_planner
            .plan_fft_forward(self.options.frame_opts.win_size_padded());
        fft.process_with_scratch(
            frame.as_slice_mut().unwrap(),
            self.fft_out.as_mut_slice(),
            self.fft_scratch.as_mut_slice(),
        )
        .unwrap();

        let spectrum = &mut self.fft_out;
        let power_spectrum = ArrayViewMut::from(Mfcc::compute_power_spectrum(spectrum));

        mel_banks.apply_in(power_spectrum.view(), self.mel_energies.view_mut());
        for energy in self.mel_energies.iter_mut() {
            if *energy == 0.0 {
                *energy = f32::EPSILON;
            }
            *energy = energy.ln();
        }

        general_mat_vec_mul(1.0, &self.dct_matrix, &self.mel_energies, 0.0, &mut feature)

        // XXX: Do cepstral liftering
    }

    fn frame_options(&self) -> &FrameExtractionOpts {
        &self.options.frame_opts
    }

    fn n_coeffs(&self) -> usize {
        self.options.n_ceps
    }
}

struct FrameExtractor<'a, T: FrameSupplier> {
    frame_supplier: &'a mut T,
    first: bool,
    opts: FrameExtractionOpts,
    last: Box<[Float]>,
    current: Box<[Float]>,
    idx: usize,
    buf: Box<[Float]>,
    buf_len: usize,
}

impl<'a, T: FrameSupplier> FrameExtractor<'a, T> {
    fn new(supplier: &'a mut T, options: FrameExtractionOpts) -> Self {
        let frame_size = options.win_size_padded();
        Self {
            frame_supplier: supplier,
            first: true,
            opts: options,
            last: vec![0.; frame_size].into_boxed_slice(),
            current: vec![0.; frame_size].into_boxed_slice(),
            idx: 0,
            buf: vec![0.; frame_size].into_boxed_slice(),
            buf_len: 0,
        }
    }

    fn extract_frame(&mut self) -> std::ops::ControlFlow<&mut [Float], &mut [Float]> {
        let len = self.opts.win_size_padded();
        let mut filled = if self.first {
            self.first = false;
            0
        } else {
            let n = len - self.opts.win_shift();
            self.current[..n].copy_from_slice(&self.last[self.opts.win_shift()..]);
            n
        };

        while filled < len {
            if self.idx >= self.buf_len {
                self.buf_len = self.frame_supplier.fill_next(&mut self.buf);
                self.idx = 0;
                if self.buf_len == 0 {
                    self.current[filled..].fill(0.);
                    return std::ops::ControlFlow::Break(&mut self.current);
                }
            }
            let from_buf = usize::min(self.buf_len - self.idx, len - filled);
            self.current[filled..filled + from_buf]
                .copy_from_slice(&self.buf[self.idx..self.idx + from_buf]);
            filled += from_buf;
            self.idx += from_buf;
        }
        self.last.copy_from_slice(&self.current);
        std::ops::ControlFlow::Continue(&mut self.current)
    }
}

pub struct MfccOptions {
    pub mel_opts: MelBanksOpts,
    pub frame_opts: FrameExtractionOpts,
    pub n_ceps: usize,
}

pub struct MfccComputer {
    feature: Mfcc,
}

fn hamming_window(len: usize) -> Array1<Float> {
    let a = TWOPI / (len - 1) as Float;
    Array1::from_iter((0..len).map(|i| 0.54 - 0.46 * f32::cos(a * i as f32)))
}

pub trait FrameSupplier {
    fn n_samples_est(&self) -> usize;
    fn fill_next(&mut self, output: &mut [Float]) -> usize;
}

impl MfccComputer {
    pub fn new(feature: Mfcc) -> Self {
        Self { feature }
    }

    fn preemphasize(frame: &mut [Float], opts: &FrameExtractionOpts) {
        let emph_fact = opts.emphasis_factor;
        for j in 1..frame.len() {
            frame[j] -= emph_fact * frame[j - 1];
        }
        frame[0] -= emph_fact * frame[0];
    }

    pub fn compute(&mut self, wave: &mut impl FrameSupplier) -> Vec<[f32; 13]> {
        let n_samples = wave.n_samples_est();
        let opts = *self.feature.frame_options();
        let frame_shift = opts.win_shift();
        let frame_length_padded = self.feature.frame_options().win_size_padded();
        let num_frames = n_samples / frame_shift + 1;

        let mut frame_extractor = FrameExtractor::new(wave, opts);
        let hamming_window = hamming_window(frame_length_padded);
        let mut output = Vec::with_capacity(num_frames);

        let mut keep_going = true;

        while keep_going {
            let frame = match frame_extractor.extract_frame() {
                std::ops::ControlFlow::Continue(f) => f,
                std::ops::ControlFlow::Break(f) => {
                    keep_going = false;
                    f
                }
            };
            Self::preemphasize(frame, &opts);
            let mut frame = ArrayViewMut1::from(frame);
            frame *= &hamming_window;

            output.push([0.; 13]);
            let feat_view = ArrayViewMut1::from(output.last_mut().unwrap());
            self.feature.compute(frame.view_mut(), feat_view);
        }
        output
    }
}
