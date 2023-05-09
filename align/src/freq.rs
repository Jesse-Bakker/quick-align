use core::slice;
use std::ops;

macro_rules! impl_float_ops {
    ($t:ty) => {
        impl ops::Mul<f32> for $t {
            type Output = $t;

            fn mul(self, rhs: f32) -> Self::Output {
                Self(self.0 * rhs)
            }
        }

        impl ops::Mul<$t> for f32 {
            type Output = $t;

            fn mul(self, rhs: $t) -> Self::Output {
                <$t>::from(self * rhs.0)
            }
        }
        impl ops::Div<f32> for $t {
            type Output = $t;

            fn div(self, rhs: f32) -> Self::Output {
                Self(self.0 / rhs)
            }
        }

        impl ops::Sub<$t> for $t {
            type Output = $t;

            fn sub(self, rhs: $t) -> Self::Output {
                Self(self.0 - rhs.0)
            }
        }
        impl ops::Add<$t> for $t {
            type Output = $t;

            fn add(self, rhs: $t) -> Self::Output {
                Self(self.0 + rhs.0)
            }
        }

        impl From<$t> for f32 {
            fn from(t: $t) -> f32 {
                t.0
            }
        }

        impl From<f32> for $t {
            fn from(f: f32) -> Self {
                Self(f)
            }
        }

        impl From<&$t> for f32 {
            fn from(t: &$t) -> f32 {
                t.0
            }
        }

        impl From<&f32> for $t {
            fn from(f: &f32) -> Self {
                Self(*f)
            }
        }
        impl Asf32Slice for [$t] {
            fn as_float_slice(&self) -> &[f32] {
                let ptr = self.as_ptr();
                let len = self.len();
                unsafe { slice::from_raw_parts(ptr.cast(), len) }
            }
        }
    };
}

#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct Freq(f32);

#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub(crate) struct MelFreq(f32);

pub(crate) trait Asf32Slice {
    fn as_float_slice(&self) -> &[f32];
}

impl_float_ops!(Freq);

impl_float_ops!(MelFreq);

impl Freq {
    pub(crate) fn to_mel(self) -> MelFreq {
        MelFreq(1127.0 * (1.0 + self.0 / 700.0).ln())
    }
}

impl MelFreq {
    pub(crate) fn to_freq(self) -> Freq {
        Freq(700.0 * ((self.0 / 1127.0).exp() - 1.0))
    }
}
