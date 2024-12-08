use std::{any::TypeId, borrow::Cow, fmt::Debug, marker::PhantomData, mem::ManuallyDrop};

use bevy::{
    animation::{AnimationEvaluationError, AnimationTargetId, RepeatAnimation},
    prelude::*,
};

pub fn lerp(start: f32, end: f32, t: f32) -> f32 {
    (1. - t) * start + end * t
}

#[derive(Debug)]
pub struct ColorAlpha<Src, F> {
    f: F,
    _marker: PhantomData<fn() -> Src>,
}

impl<Src, F: Clone> Clone for ColorAlpha<Src, F> {
    fn clone(&self) -> Self {
        ColorAlpha {
            f: self.f.clone(),
            _marker: PhantomData,
        }
    }
}

impl<Src, F: Fn(&mut Src) -> &mut Color> ColorAlpha<Src, F> {
    pub fn new(f: F) -> Self {
        ColorAlpha {
            f,
            _marker: PhantomData,
        }
    }
}

impl<Src: Component, F: Send + Sync + 'static + Fn(&mut Src) -> &mut Color> AnimatableProperty
    for ColorAlpha<Src, F>
{
    type Property = f32;
    fn get_mut<'a>(
        &self,
        entity: &'a mut bevy::animation::AnimationEntityMut,
    ) -> Result<&'a mut Self::Property, AnimationEvaluationError> {
        let color_src =
            entity
                .get_mut::<Src>()
                .ok_or(AnimationEvaluationError::ComponentNotPresent(TypeId::of::<
                    BackgroundColor,
                >(
                )))?;
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
            | Color::Xyza(Xyza { alpha, .. }) => Ok(alpha),
        }
    }
    fn evaluator_id(&self) -> EvaluatorId {
        EvaluatorId::Type(TypeId::of::<Self>())
    }
}

/// Makes a very simple alpha fade animation. This is not the ideal way to do things but
/// it does cut down on boilerplate for my use case, which has a decent amount of alpha
/// fading. Don't treat this like a hammer if you don't have many nails.
///
/// Defaults to setting the player to never repeat.
pub fn alpha_fade_animation<Src: Component, E>(
    start: f32,
    end: f32,
    duration: f32,
    extract_alpha: E,
    target_name: impl Into<Cow<'static, str>>,
    animations: &mut Assets<AnimationClip>,
    modify_clip: impl FnOnce(&mut AnimationClip),
) -> (AnimationTargetId, AnimationPlayer, AnimationGraph)
where
    E: Clone + Send + Sync + 'static + Fn(&mut Src) -> &mut Color,
{
    let target = AnimationTargetId::from_name(&Name::new(target_name));
    let mut clip = AnimationClip::default();
    clip.add_curve_to_target(
        target,
        AnimatableCurve::new(
            ColorAlpha::new(extract_alpha),
            EasingCurve::new(start, end, EaseFunction::Linear)
                .reparametrize_linear(Interval::new(0., duration).unwrap())
                .unwrap(),
        ),
    );
    modify_clip(&mut clip);
    let (graph, index) = AnimationGraph::from_clip(animations.add(clip));
    let mut anim_player = AnimationPlayer::default();
    anim_player.play(index).set_repeat(RepeatAnimation::Never);
    (target, anim_player, graph)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Reflect)]
pub struct StatePair<S1, S2>(pub S1, pub S2);

impl<S1: States, S2: States> ComputedStates for StatePair<S1, S2> {
    type SourceStates = (S1, S2);
    fn compute(sources: Self::SourceStates) -> Option<Self> {
        Some(StatePair(sources.0, sources.1))
    }
}

// i didn't find an implementation of this i liked in a quick search and it's really easy to write so i just wrote it
pub struct DropGuard<F: FnOnce()> {
    f: ManuallyDrop<F>,
}

impl<F: FnOnce()> Drop for DropGuard<F> {
    fn drop(&mut self) {
        // SAFETY: self.f is not used again after self is dropped
        unsafe { ManuallyDrop::take(&mut self.f)() }
    }
}

pub fn drop_guard<F: FnOnce()>(f: F) -> DropGuard<F> {
    DropGuard {
        f: ManuallyDrop::new(f),
    }
}
