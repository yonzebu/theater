use std::{fmt::Debug, sync::Arc};

use bevy::{math::VectorSpace, prelude::*};

pub fn lerp(start: f32, end: f32, t: f32) -> f32 {
    (1. - t) * start + end * t
}

pub trait DynCurveChainCompatible<T>: Curve<T> + Debug + 'static + Send + Sync {}
impl<T, C> DynCurveChainCompatible<T> for C where C: Curve<T> + Debug + 'static + Send + Sync {}

/// Curves are composed as if with chain_continue. This is an ugly hack.
#[derive(Clone, Debug, Default, Reflect)]
pub struct DynCurveChain<T> {
    #[reflect(ignore)]
    current_curve: Option<Arc<dyn DynCurveChainCompatible<T>>>,
}

impl<T: VectorSpace + 'static + Send + Sync> DynCurveChain<T> {
    pub fn new() -> Self {
        DynCurveChain {
            current_curve: None,
        }
    }

    pub fn append_curve(&mut self, curve: impl DynCurveChainCompatible<T>) {
        if let Some(old_curve) = self.current_curve.take() {
            self.current_curve = Some(Arc::new(old_curve.chain_continue(curve).unwrap()));
        } else {
            self.current_curve = Some(Arc::new(curve))
        }
    }
}

impl<T: TypePath> Curve<T> for DynCurveChain<T> {
    fn domain(&self) -> Interval {
        if let Some(curve) = self.current_curve.as_ref() {
            curve.domain()
        } else {
            Interval::EVERYWHERE
        }
    }
    fn sample(&self, t: f32) -> Option<T> {
        self.current_curve
            .as_ref()
            .and_then(|curve| curve.sample(t))
    }
    fn sample_unchecked(&self, t: f32) -> T {
        self.current_curve.as_ref().unwrap().sample_unchecked(t)
    }
}
