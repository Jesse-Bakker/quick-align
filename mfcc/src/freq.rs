use super::Float;
use core::slice;
use std::ops;

macro_rules! impl_float_ops {
    ($t:ty) => {
        impl ops::Mul<Float> for $t {
            type Output = $t;

            fn mul(self, rhs: Float) -> Self::Output {
                Self(self.0 * rhs)
            }
        }

        impl ops::Mul<$t> for Float {
            type Output = $t;

            fn mul(self, rhs: $t) -> Self::Output {
                <$t>::from(self * rhs.0)
            }
        }
        impl ops::Div<Float> for $t {
            type Output = $t;

            fn div(self, rhs: Float) -> Self::Output {
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

        impl From<$t> for Float {
            fn from(t: $t) -> Float {
                t.0
            }
        }

        impl From<Float> for $t {
            fn from(f: Float) -> Self {
                Self(f)
            }
        }

        impl From<&$t> for Float {
            fn from(t: &$t) -> Float {
                t.0
            }
        }

        impl From<&Float> for $t {
            fn from(f: &Float) -> Self {
                Self(*f)
            }
        }
        impl AsFloatSlice for [$t] {
            fn as_float_slice(&self) -> &[Float] {
                let ptr = self.as_ptr();
                let len = self.len();
                unsafe { slice::from_raw_parts(ptr.cast(), len) }
            }
        }
    };
}

#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct Freq(Float);

#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub(crate) struct MelFreq(Float);

pub(crate) trait AsFloatSlice {
    fn as_float_slice(&self) -> &[Float];
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
