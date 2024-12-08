use std::{any::TypeId, fmt::Debug, marker::PhantomData, sync::Arc};

use bevy::{animation::AnimationEvaluationError, math::VectorSpace, prelude::*};

pub fn lerp(start: f32, end: f32, t: f32) -> f32 {
    (1. - t) * start + end * t
}

#[derive(Clone, Debug)]
pub struct ColorAlpha<Src, F> {
    f: F,
    _marker: PhantomData<fn() -> Src>
}

impl<Src, F: for<'a> Fn(&'a mut Src) -> &'a mut Color> ColorAlpha<Src, F> {
    pub fn new(f: F) -> Self {
        ColorAlpha {
            f,
            _marker: PhantomData
        }
    }
}

impl<Src: Component, F: Send + Sync + 'static + for<'a> Fn(&'a mut Src) -> &'a mut Color> AnimatableProperty for ColorAlpha<Src, F> {
    type Property = f32;
    fn get_mut<'a>(
        &self,
        entity: &'a mut bevy::animation::AnimationEntityMut,
    ) -> Result<&'a mut Self::Property, AnimationEvaluationError> {
        let mut color_src = entity
            .get_mut::<Src>()
            .ok_or(AnimationEvaluationError::ComponentNotPresent(TypeId::of::<BackgroundColor>()))?;
        match (self.f)(color_src.into_inner()) {
            Color::Hsla(Hsla { alpha, .. }) 
            | Color::Hsva(Hsva { alpha, .. }) 
            | Color::Hwba(Hwba { alpha, .. })
            | Color::Laba(Laba { alpha, .. })
            | Color::Lcha(Lcha { alpha, .. })
            | Color::LinearRgba(LinearRgba { alpha, .. })
            | Color::Oklaba(Oklaba { alpha, .. })
            | Color::Oklcha(Oklcha { alpha, .. })
            | Color::Srgba(Srgba { alpha, .. })
            | Color::Xyza(Xyza { alpha, .. }) => Ok(alpha)
        }
    }
    fn evaluator_id(&self) -> EvaluatorId {
        EvaluatorId::Type(TypeId::of::<Self>())
    }
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
