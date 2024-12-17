#![allow(clippy::type_complexity, clippy::too_many_arguments)]
use core::fmt::Debug;

use bevy::animation::{animated_field, AnimationTarget, AnimationTargetId, RepeatAnimation};
use bevy::asset::{RenderAssetUsages, UntypedAssetId};
use bevy::audio::PlaybackMode;
use bevy::ecs::component::ComponentId;
use bevy::ecs::world::DeferredWorld;
use bevy::pbr::{ExtendedMaterial, MaterialExtension, NotShadowCaster};
use bevy::render::render_resource::{
    AsBindGroup, Extent3d, Face, ShaderRef, TextureDimension, TextureFormat,
};
use bevy::utils::hashbrown::HashMap;
use bevy::{math::vec3, prelude::*};

mod debug;
use debug::*;
use screen_light::{ScreenLight, ScreenLightExtension, ScreenLightPlugin};
use script::{RunnerUpdated, ScriptChoices, ScriptPlugin, ScriptRunner, UpdateRunner};
use video::{
    VideoFinished, VideoPlayer, VideoPlayerPlugin, VideoPlugin, VideoStream, VideoStreamSettings,
};
mod screen_light;
mod script;
mod util;
mod video;
use util::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Event)]
struct AnimationFinished;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, States)]
enum Progress {
    #[default]
    Start,
    // these two states are disgusting, there's probably something nicer to be made by using the animation graph system or something
    // on the other hand, that's orthogonal to states, so maybe not?
    // the tldr of why they're needed is that there are several time blocks during
    // which loading could finish and different behaviors that are desired for each time block
    FadeOutStartText,
    FadeIn,
    Entering,
    Preshow,
    Show,
    Postshow,
    Leaving,
    End,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, States, Deref, DerefMut)]
struct Loaded(bool);

#[derive(Clone, Asset, TypePath, AsBindGroup)]
struct Paper {}

impl MaterialExtension for Paper {
    fn vertex_shader() -> ShaderRef {
        "paper.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "paper.wgsl".into()
    }
}

#[derive(Component)]
#[component(storage = "SparseSet")]
struct Chair;

#[derive(Component)]
#[component(storage = "SparseSet")]
struct Screen;

#[derive(Component)]
#[component(storage = "SparseSet")]
struct UiRoot;

#[derive(Component)]
#[component(storage = "SparseSet")]
struct TheaterUiRoot;

/// The thing that controls the size of the text box by being a bounding rectangle
#[derive(Component)]
#[component(storage = "SparseSet")]
struct TextBoxWrapper;

#[derive(Component)]
struct Instructions;

#[derive(Component)]
#[component(storage = "SparseSet")]
struct StartUiRoot;

#[derive(Component)]
#[component(storage = "SparseSet")]
struct StartText;

#[derive(Component)]
#[component(storage = "SparseSet")]
struct LoadingText;

#[derive(Component)]
#[component(storage = "SparseSet")]
struct You;

#[derive(Component)]
#[component(storage = "SparseSet")]
struct Me;

const CHAIR_ROWS: i32 = 5;
const CHAIR_COLS: i32 = 5;
const SCREEN_POS: f32 = -9.9;
const SCREEN_WIDTH: f32 = 18.;
const SCREEN_HEIGHT: f32 = SCREEN_WIDTH * 9. / 16.;
const SCREEN_LIGHT_POS: f32 = SCREEN_POS - 10. / 4.;

#[derive(Resource, Deref, DerefMut)]
struct AssetsToLoad(Vec<UntypedAssetId>);

impl<I> From<I> for AssetsToLoad
where
    I: IntoIterator,
    I::Item: Into<UntypedAssetId>,
{
    fn from(value: I) -> Self {
        AssetsToLoad(value.into_iter().map(|item| item.into()).collect())
    }
}

#[derive(Component)]
#[component(on_add = WaitingForLoads::on_add)]
#[component(on_remove = WaitingForLoads::on_remove)]
struct WaitingForLoads;

impl WaitingForLoads {
    fn on_add(mut deferred_world: DeferredWorld, entity: Entity, _: ComponentId) {
        if let Some(mut visibility) = deferred_world.get_mut::<Visibility>(entity) {
            *visibility = Visibility::Hidden;
        } else {
            deferred_world
                .commands()
                .entity(entity)
                .insert(Visibility::Hidden);
        }
    }
    fn on_remove(mut deferred_world: DeferredWorld, entity: Entity, _: ComponentId) {
        if let Some(mut visibility) = deferred_world.get_mut::<Visibility>(entity) {
            *visibility = Visibility::Inherited;
        }
    }
}

#[derive(Debug, Component, Deref, DerefMut)]
struct FutureVideoPlayer(Handle<VideoStream>);

#[derive(Debug, Resource, Deref, DerefMut)]
struct ScreenOffImage(Handle<Image>);

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut papers: ResMut<Assets<ExtendedMaterial<StandardMaterial, Paper>>>,
    mut images: ResMut<Assets<Image>>,
    mut animations: ResMut<Assets<AnimationClip>>,
    mut anim_graphs: ResMut<Assets<AnimationGraph>>,
    assets: Res<AssetServer>,
) {
    commands.spawn((Camera3d::default(), Transform::from_xyz(0., 2., 5.)));
    commands.add_observer(on_start_show);
    commands.add_observer(on_video_finished);
    let video_stream =
        assets.load_with_settings("show.mp4", |settings: &mut VideoStreamSettings| {
            // mips are too slow to use and i don't have time/want to improve them right now
            settings.use_mips = false;
        });
    let screen_off_image = images.add(Image::new(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        vec![200, 200, 200, 0],
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::all(),
    ));
    commands.spawn((
        AudioPlayer(video_stream.clone()),
        PlaybackSettings {
            mode: PlaybackMode::Once,
            paused: true,
            ..default()
        },
    ));
    commands.spawn((
        // keep transform synced with screen transform
        Transform::from_xyz(0., 2.5, SCREEN_LIGHT_POS).looking_to(Dir3::Z, Dir3::Y),
        ScreenLight {
            image: screen_off_image.clone(),
        },
        FutureVideoPlayer(video_stream.clone()),
        Projection::Perspective(PerspectiveProjection {
            fov: 2. * f32::atan(SCREEN_HEIGHT / 2. / (SCREEN_POS - SCREEN_LIGHT_POS)),
            aspect_ratio: 16. / 9.,
            near: SCREEN_POS - SCREEN_LIGHT_POS,
            ..default()
        }),
    ));

    let rectangle = meshes.add(Rectangle::new(1.0, 1.0));
    let black = materials.add(Color::linear_rgb(0., 0., 0.));
    // screen
    commands.spawn((
        Mesh3d(rectangle.clone()),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color_texture: Some(screen_off_image.clone()),
            ..default()
        })),
        Transform {
            translation: vec3(0., 2.5, SCREEN_POS),
            scale: vec3(SCREEN_WIDTH, SCREEN_HEIGHT, 1.),
            ..default()
        },
        FutureVideoPlayer(video_stream.clone()),
        WaitingForLoads,
        NotShadowCaster,
        Screen,
    ));
    // screen border
    commands.spawn((
        Mesh3d(rectangle.clone()),
        MeshMaterial3d(black.clone()),
        Transform {
            translation: vec3(0., 2.5, SCREEN_POS * 1.01),
            scale: vec3(18.5, 0.5 + 18. * 9. / 16., 1.),
            ..default()
        },
        WaitingForLoads,
        NotShadowCaster,
    ));
    // screen wall
    commands.spawn((
        Mesh3d(rectangle.clone()),
        MeshMaterial3d(materials.add(Color::linear_rgb(0.1, 0.1, 0.1))),
        Transform {
            translation: vec3(0., 0., -10.),
            scale: vec3(30., 30., 10.),
            ..default()
        },
        WaitingForLoads,
        NotShadowCaster,
    ));

    // other walls
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::from_size(vec3(20., 20., 15.)).mesh().build())),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::linear_rgb(0.5, 0.125, 0.125),
            cull_mode: Some(Face::Front),
            ..default()
        })),
        Transform::from_xyz(0., 0., -5.),
        WaitingForLoads,
        NotShadowCaster,
    ));

    const STEP_UP_EASING: EaseFunction = EaseFunction::QuadraticIn;
    const STEP_DOWN_EASING: EaseFunction = EaseFunction::QuadraticOut;
    let watching_rot = Transform::default().looking_to(Dir3::Y, Dir3::Z).rotation;
    let enter_rot = Quat::from_rotation_y(f32::to_radians(45.)) * watching_rot;
    let look_right_rot = Quat::from_rotation_y(f32::to_radians(-45.)) * watching_rot;
    let look_left_rot = Quat::from_rotation_y(f32::to_radians(45.)) * watching_rot;

    let mut leaving_clip = AnimationClip::default();
    let me_target_id = AnimationTargetId::from_name(&Name::new("me"));
    // translation (very ugly, there aren't many good dynamic ways to construct curves right now i don't think)
    leaving_clip.add_curve_to_target(
        me_target_id,
        AnimatableCurve::new(
            animated_field!(Transform::translation),
            // stand up
            EasingCurve::new(
                vec3(-2., 0., -0.5),
                vec3(-2., 1., -0.75),
                EaseFunction::ExponentialOut,
            )
            .reparametrize_linear(Interval::new(0., 0.25).unwrap())
            .unwrap()
            // stay still (start rotating at 0.5)
            .chain(ConstantCurve::new(
                Interval::new(0.25, 1.).unwrap(),
                vec3(-2., 1., -0.75),
            ))
            .unwrap()
            .chain(
                // step 1 up
                EasingCurve::new(vec3(-2., 1., -0.75), vec3(-2.4, 1.1, -0.75), STEP_UP_EASING)
                    .reparametrize_linear(Interval::new(1., 1.5).unwrap())
                    .unwrap(),
            )
            .unwrap()
            // step 1 down
            .chain(
                EasingCurve::new(
                    vec3(-2.4, 1.1, -0.75),
                    vec3(-2.8, 1., -0.75),
                    STEP_DOWN_EASING,
                )
                .reparametrize_linear(Interval::new(1.5, 2.).unwrap())
                .unwrap(),
            )
            .unwrap()
            // step 2 up
            .chain(
                EasingCurve::new(
                    vec3(-2.8, 1., -0.75),
                    vec3(-3.2, 1.1, -0.75),
                    STEP_UP_EASING,
                )
                .reparametrize_linear(Interval::new(2., 2.5).unwrap())
                .unwrap(),
            )
            .unwrap()
            // step 2 down
            .chain(
                EasingCurve::new(
                    vec3(-3.2, 1.1, -0.75),
                    vec3(-3.6, 1., -0.75),
                    STEP_DOWN_EASING,
                )
                .reparametrize_linear(Interval::new(2.5, 3.).unwrap())
                .unwrap(),
            )
            .unwrap()
            // step 3 up
            .chain(
                EasingCurve::new(vec3(-3.6, 1., -0.75), vec3(-4., 1.1, -0.75), STEP_UP_EASING)
                    .reparametrize_linear(Interval::new(3., 3.5).unwrap())
                    .unwrap(),
            )
            .unwrap()
            // step 3 down
            .chain(
                EasingCurve::new(
                    vec3(-4., 1.1, -0.75),
                    vec3(-4.4, 1., -0.75),
                    STEP_DOWN_EASING,
                )
                .reparametrize_linear(Interval::new(3.5, 4.).unwrap())
                .unwrap(),
            )
            .unwrap()
            // step 4 up
            .chain(
                EasingCurve::new(
                    vec3(-4.4, 1., -0.75),
                    vec3(-4.8, 1.1, -0.75),
                    STEP_UP_EASING,
                )
                .reparametrize_linear(Interval::new(4., 4.5).unwrap())
                .unwrap(),
            )
            .unwrap()
            // step 4 down
            .chain(
                EasingCurve::new(
                    vec3(-4.8, 1.1, -0.75),
                    vec3(-5.2, 1., -0.75),
                    STEP_DOWN_EASING,
                )
                .reparametrize_linear(Interval::new(4.5, 5.).unwrap())
                .unwrap(),
            )
            .unwrap()
            // step 5 up
            .chain(
                EasingCurve::new(
                    vec3(-5.2, 1., -0.75),
                    vec3(-5.6, 1.1, -0.75),
                    STEP_UP_EASING,
                )
                .reparametrize_linear(Interval::new(5., 5.5).unwrap())
                .unwrap(),
            )
            .unwrap()
            // step 5 down
            .chain(
                EasingCurve::new(
                    vec3(-5.6, 1.1, -0.75),
                    vec3(-6., 1., -0.75),
                    STEP_DOWN_EASING,
                )
                .reparametrize_linear(Interval::new(5.5, 6.).unwrap())
                .unwrap(),
            )
            .unwrap(),
        ),
    );
    // all rotation in the animation
    leaving_clip.add_curve_to_target(
        me_target_id,
        AnimatableCurve::new(
            animated_field!(Transform::rotation),
            // steps
            ConstantCurve::new(Interval::new(0., 0.5).unwrap(), watching_rot)
                // look left
                .chain(
                    EasingCurve::new(enter_rot, look_left_rot, EaseFunction::BackOut)
                        .reparametrize_linear(Interval::new(0.5, 1.).unwrap())
                        .unwrap(),
                )
                .unwrap(),
        ),
    );
    leaving_clip.add_event(6., AnimationFinished);
    let (leaving_graph, leaving_index) = AnimationGraph::from_clip(animations.add(leaving_clip));
    let mut me_anim_player = AnimationPlayer::default();
    me_anim_player
        .play(leaving_index)
        .set_repeat(RepeatAnimation::Never)
        .pause();

    let me_image = assets.load("me.png");
    let me_mesh = assets.load("me.glb#Mesh0/Primitive0");
    let me_entity = commands
        .spawn((
            Mesh3d(me_mesh.clone()),
            MeshMaterial3d(papers.add(ExtendedMaterial {
                base: StandardMaterial::from(me_image.clone()),
                extension: Paper {},
            })),
            Transform::from_xyz(-2., 0., -0.5)
                .looking_to(Dir3::Y, Dir3::Z)
                .with_scale(Vec3::ONE * 0.5),
            WaitingForLoads,
            Me,
        ))
        .id();
    commands
        .entity(me_entity)
        .insert((
            AnimationTarget {
                id: me_target_id,
                player: me_entity,
            },
            me_anim_player,
            AnimationGraphHandle(anim_graphs.add(leaving_graph)),
        ))
        .observe(on_leave_animation_finished);

    let mut entering_clip = AnimationClip::default();
    let you_target_id = AnimationTargetId::from_name(&Name::new("you"));
    // translation (very ugly, there aren't many good dynamic ways to construct curves right now i don't think)
    entering_clip.add_curve_to_target(
        you_target_id,
        AnimatableCurve::new(
            animated_field!(Transform::translation),
            // step 1 up
            EasingCurve::new(vec3(6., 1., -0.75), vec3(5.6, 1.1, -0.75), STEP_UP_EASING)
                .reparametrize_linear(Interval::new(0., 0.5).unwrap())
                .unwrap()
                // step 1 down
                .chain(
                    EasingCurve::new(
                        vec3(5.6, 1.1, -0.75),
                        vec3(5.2, 1., -0.75),
                        STEP_DOWN_EASING,
                    )
                    .reparametrize_linear(Interval::new(0.5, 1.).unwrap())
                    .unwrap(),
                )
                .unwrap()
                // step 2 up
                .chain(
                    EasingCurve::new(vec3(5.2, 1., -0.75), vec3(4.8, 1.1, -0.75), STEP_UP_EASING)
                        .reparametrize_linear(Interval::new(1., 1.5).unwrap())
                        .unwrap(),
                )
                .unwrap()
                // step 2 down
                .chain(
                    EasingCurve::new(
                        vec3(4.8, 1.1, -0.75),
                        vec3(4.4, 1., -0.75),
                        STEP_DOWN_EASING,
                    )
                    .reparametrize_linear(Interval::new(1.5, 2.).unwrap())
                    .unwrap(),
                )
                .unwrap()
                // step 3 up
                .chain(
                    EasingCurve::new(vec3(4.4, 1., -0.75), vec3(4., 1.1, -0.75), STEP_UP_EASING)
                        .reparametrize_linear(Interval::new(2., 2.5).unwrap())
                        .unwrap(),
                )
                .unwrap()
                // step 3 down
                .chain(
                    EasingCurve::new(vec3(4., 1.1, -0.75), vec3(3.6, 1., -0.75), STEP_DOWN_EASING)
                        .reparametrize_linear(Interval::new(2.5, 3.).unwrap())
                        .unwrap(),
                )
                .unwrap()
                // step 4 up
                .chain(
                    EasingCurve::new(vec3(3.6, 1., -0.75), vec3(3.2, 1.1, -0.75), STEP_UP_EASING)
                        .reparametrize_linear(Interval::new(3., 3.5).unwrap())
                        .unwrap(),
                )
                .unwrap()
                // step 4 down
                .chain(
                    EasingCurve::new(
                        vec3(3.2, 1.1, -0.75),
                        vec3(2.8, 1., -0.75),
                        STEP_DOWN_EASING,
                    )
                    .reparametrize_linear(Interval::new(3.5, 4.).unwrap())
                    .unwrap(),
                )
                .unwrap()
                // step 5 up
                .chain(
                    EasingCurve::new(vec3(2.8, 1., -0.75), vec3(2.4, 1.1, -0.75), STEP_UP_EASING)
                        .reparametrize_linear(Interval::new(4., 4.5).unwrap())
                        .unwrap(),
                )
                .unwrap()
                // step 5 down
                .chain(
                    EasingCurve::new(vec3(2.4, 1.1, -0.75), vec3(2., 1., -0.75), STEP_DOWN_EASING)
                        .reparametrize_linear(Interval::new(4.5, 5.).unwrap())
                        .unwrap(),
                )
                .unwrap()
                // stay still
                .chain(ConstantCurve::new(
                    Interval::new(5., 8.25).unwrap(),
                    vec3(2., 1., -0.75),
                ))
                .unwrap()
                // sit down
                .chain(
                    EasingCurve::new(
                        vec3(2., 1., -0.75),
                        vec3(2., 0., -0.5),
                        EaseFunction::ExponentialOut,
                    )
                    .reparametrize_linear(Interval::new(8.25, 8.5).unwrap())
                    .unwrap(),
                )
                .unwrap(),
        ),
    );
    // all rotation in the animation
    entering_clip.add_curve_to_target(
        you_target_id,
        AnimatableCurve::new(
            animated_field!(Transform::rotation),
            // steps
            ConstantCurve::new(Interval::new(0., 5.5).unwrap(), enter_rot)
                // look right
                .chain(
                    EasingCurve::new(enter_rot, look_right_rot, EaseFunction::BackOut)
                        .reparametrize_linear(Interval::new(5.5, 6.).unwrap())
                        .unwrap(),
                )
                .unwrap()
                // pause
                .chain(ConstantCurve::new(
                    Interval::new(6., 6.5).unwrap(),
                    look_right_rot,
                ))
                .unwrap()
                // look left
                .chain(
                    EasingCurve::new(look_right_rot, look_left_rot, EaseFunction::BackOut)
                        .reparametrize_linear(Interval::new(6.5, 7.).unwrap())
                        .unwrap(),
                )
                .unwrap()
                // pause
                .chain(ConstantCurve::new(
                    Interval::new(7., 7.5).unwrap(),
                    look_left_rot,
                ))
                .unwrap()
                // look forward
                .chain(
                    EasingCurve::new(look_left_rot, watching_rot, EaseFunction::BackOut)
                        .reparametrize_linear(Interval::new(7.5, 8.).unwrap())
                        .unwrap(),
                )
                .unwrap(),
        ),
    );
    entering_clip.add_event(8.5, AnimationFinished);
    let (entering_graph, entering_index) = AnimationGraph::from_clip(animations.add(entering_clip));
    let mut you_anim_player = AnimationPlayer::default();
    you_anim_player
        .play(entering_index)
        .set_repeat(RepeatAnimation::Never)
        .pause();
    let you_image = assets.load("you.png");
    let you_mesh = assets.load("you.glb#Mesh0/Primitive0");
    let you_entity = commands
        .spawn((
            Mesh3d(you_mesh.clone()),
            MeshMaterial3d(papers.add(ExtendedMaterial {
                base: StandardMaterial::from(you_image.clone()),
                extension: Paper {},
            })),
            Transform::from_xyz(6., 1., -0.5)
                .looking_to(Dir3::Y, Dir3::Z)
                .with_scale(Vec3::ONE * 0.5),
            WaitingForLoads,
            You,
            DebugMarker,
        ))
        .id();
    commands
        .entity(you_entity)
        .insert((
            AnimationTarget {
                id: you_target_id,
                player: you_entity,
            },
            you_anim_player,
            AnimationGraphHandle(anim_graphs.add(entering_graph)),
        ))
        .observe(on_enter_animation_finished);

    // chairs
    let chair: Handle<Scene> = assets.load("chair.glb#Scene0");
    for i in 0i32..CHAIR_COLS {
        for j in 0i32..CHAIR_ROWS {
            let x = 2. * (i - 2) as f32;
            let y = -j as f32 * 0.75;
            let z = -j as f32 * 2.;
            commands.spawn((
                SceneRoot(chair.clone()),
                Transform {
                    translation: vec3(x, y, z),
                    scale: Vec3::ONE * 0.25,
                    ..default()
                },
                DebugMarker,
                Chair,
                WaitingForLoads,
            ));
        }
    }

    let script = assets.load("script.txt");
    let script_choices_entity = commands.spawn_empty().id();
    let mut script_runner = ScriptRunner::new(script.clone(), script_choices_entity, 15.0);
    script_runner.pause();
    let root = commands
        .spawn((
            Node {
                align_self: AlignSelf::Stretch,
                justify_self: JustifySelf::Stretch,
                flex_wrap: FlexWrap::NoWrap,
                justify_content: JustifyContent::Stretch,
                align_items: AlignItems::Stretch,
                align_content: AlignContent::Stretch,
                width: Val::Vw(100.),
                height: Val::Vh(100.),
                ..default()
            },
            UiRoot,
        ))
        .id();

    // start ui
    let start_screen = commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                flex_direction: FlexDirection::Column,
                align_self: AlignSelf::Stretch,
                justify_self: JustifySelf::Stretch,
                justify_content: JustifyContent::Center,
                justify_items: JustifyItems::Center,
                align_content: AlignContent::Center,
                align_items: AlignItems::Center,
                width: Val::Vw(100.),
                height: Val::Vh(100.),
                ..default()
            },
            BackgroundColor(Color::linear_rgb(0., 0., 0.)),
            Visibility::Inherited,
            StartUiRoot,
        ))
        .id();
    let start_text = commands
        .spawn((
            Node {
                align_self: AlignSelf::Center,
                justify_self: JustifySelf::Center,
                ..default()
            },
            Text("this game is controlled with the mouse\n\n(click anywhere to continue)".into()),
            TextLayout {
                justify: JustifyText::Center,
                ..default()
            },
            TextFont {
                font_size: 30.,
                ..default()
            },
            StartText,
        ))
        .id();
    commands.entity(start_screen).observe(on_start_clicked);
    commands.entity(start_text).observe(on_start_clicked);
    commands.entity(root).add_child(start_screen);
    commands.entity(start_screen).add_child(start_text);

    // theater ui
    let theater_root = commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                flex_direction: FlexDirection::ColumnReverse,
                align_self: AlignSelf::Stretch,
                justify_self: JustifySelf::Stretch,
                flex_wrap: FlexWrap::NoWrap,
                justify_content: JustifyContent::FlexStart,
                align_items: AlignItems::End,
                align_content: AlignContent::Center,
                width: Val::Vw(100.),
                height: Val::Vh(100.),
                padding: UiRect::horizontal(Val::VMin(30.)).with_bottom(Val::VMin(1.)),
                ..default()
            },
            Visibility::Hidden,
            TheaterUiRoot,
        ))
        .id();

    let text_box_wrapper = commands
        .spawn((
            Node {
                align_self: AlignSelf::Stretch,
                height: Val::Percent(20.),
                margin: UiRect::horizontal(Val::VMin(5.)),
                flex_direction: FlexDirection::Column,
                ..default()
            },
            Visibility::Hidden,
            TextBoxWrapper,
        ))
        .id();
    let instructions_text = commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                align_self: AlignSelf::Center,
                justify_self: JustifySelf::End,
                bottom: Val::Percent(10.),
                ..default()
            },
            Text("click to continue".into()),
            TextFont {
                font_size: 15.,
                ..default()
            },
            Instructions,
        ))
        .id();
    let text_visible_box = commands
        .spawn((
            Node {
                display: Display::Flex,
                border: UiRect::all(Val::VMin(1.5)),
                width: Val::Percent(100.),
                height: Val::Percent(100.),
                padding: UiRect::all(Val::VMin(1.)),
                ..default()
            },
            BorderRadius::all(Val::Percent(20.)),
            BorderColor(Color::srgb(0.99, 0.69, 1.)),
            BackgroundColor(Color::srgba(0.8, 0.43, 1., 0.5)),
        ))
        .observe(on_text_visible_box_clicked)
        .id();
    let script_runner = commands
        .spawn((
            Node {
                align_self: AlignSelf::Stretch,
                justify_self: JustifySelf::Stretch,
                ..default()
            },
            script_runner,
        ))
        .id();
    commands.entity(root).add_child(theater_root);
    commands.entity(theater_root).add_child(text_box_wrapper);
    commands
        .entity(text_box_wrapper)
        .add_child(text_visible_box)
        .add_child(instructions_text);
    commands.entity(text_visible_box).add_child(script_runner);
    commands
        .entity(script_runner)
        .observe(on_runner_updated_update_ui)
        .observe(hide_instructions_when_next_line_not_ignored);

    let choice_box_wrapper = commands
        .spawn((Node {
            display: Display::Flex,
            margin: UiRect::bottom(Val::Vh(5.)),
            ..default()
        },))
        .id();
    commands.entity(script_choices_entity).insert((
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            border: UiRect::all(Val::VMin(0.75)),
            width: Val::Percent(100.),
            height: Val::Percent(100.),
            padding: UiRect::all(Val::VMin(1.)),
            ..default()
        },
        BorderRadius::all(Val::Percent(20.)),
        BorderColor(Color::srgb(0.99, 0.69, 1.)),
        BackgroundColor(Color::srgba(0.8, 0.43, 1., 0.5)),
        ScriptChoices::new(script_runner, choice_box_wrapper),
    ));
    commands.entity(theater_root).add_child(choice_box_wrapper);
    commands
        .entity(choice_box_wrapper)
        .add_child(script_choices_entity);

    commands.insert_resource(AssetsToLoad(vec![
        me_image.id().untyped(),
        me_mesh.id().untyped(),
        you_image.id().untyped(),
        you_mesh.id().untyped(),
        chair.id().untyped(),
        script.id().untyped(),
    ]));
    commands.insert_resource(ScreenOffImage(screen_off_image));
}

fn on_start_clicked(
    _trigger: Trigger<Pointer<Click>>,
    mut commands: Commands,
    start_text: Query<Entity, With<StartText>>,
    mut animations: ResMut<Assets<AnimationClip>>,
    mut anim_graphs: ResMut<Assets<AnimationGraph>>,
    mut next_progress: ResMut<NextState<Progress>>,
    mut was_clicked: Local<bool>,
) {
    if *was_clicked {
        return;
    }
    *was_clicked = true;
    let Ok(start_text) = start_text.get_single() else {
        return;
    };
    let (target, player, graph) = alpha_fade_animation(
        1.,
        0.,
        1.,
        |text_color: &mut TextColor| &mut text_color.0,
        "start_text",
        &mut animations,
        |clip| clip.add_event(2., AnimationFinished),
    );
    commands
        .entity(start_text)
        .insert((
            AnimationTarget {
                id: target,
                player: start_text,
            },
            player,
            AnimationGraphHandle(anim_graphs.add(graph)),
        ))
        .observe(on_start_text_fade_out_finished);
    next_progress.set(Progress::FadeOutStartText);
}

// spawns the loading text if not loaded, otherwise starts fading in to the theater
fn on_start_text_fade_out_finished(
    trigger: Trigger<AnimationFinished>,
    mut commands: Commands,
    start_ui_root: Single<(Entity, &mut Node), With<StartUiRoot>>,
    loaded: Res<State<Loaded>>,
    progress: Res<State<Progress>>,
    mut next_progress: ResMut<NextState<Progress>>,
) {
    let (start_ui_root, mut start_ui_node) = start_ui_root.into_inner();
    start_ui_node.justify_content = JustifyContent::End;
    start_ui_node.align_items = AlignItems::End;
    commands.entity(trigger.entity()).despawn_recursive();
    debug_assert_eq!(progress.get(), &Progress::FadeOutStartText);

    if !**loaded.get() {
        commands
            .spawn((
                Node {
                    align_self: AlignSelf::End,
                    justify_self: JustifySelf::End,
                    ..default()
                },
                Text("loading...".into()),
                TextLayout {
                    justify: JustifyText::Right,
                    ..default()
                },
                TextColor(Color::linear_rgb(1., 1., 1.)),
                TextFont {
                    font_size: 20.,
                    ..default()
                },
                LoadingText,
            ))
            .set_parent(start_ui_root);
    }
    next_progress.set(Progress::FadeIn);
}

fn hide_instructions_when_next_line_not_ignored(
    trigger: Trigger<RunnerUpdated>,
    mut instructions: Query<&mut Visibility, With<Instructions>>,
) {
    if let RunnerUpdated::Received {
        update: UpdateRunner::NextLine,
        ignored: false,
    } = trigger.event()
    {
        for mut visibility in instructions.iter_mut() {
            *visibility = Visibility::Hidden;
        }
    }
}

fn on_text_visible_box_clicked(
    trigger: Trigger<Pointer<Click>>,
    mut commands: Commands,
    with_children: Query<&Children>,
    runners: Query<(Entity, &ScriptRunner)>,
) {
    if let Some((runner_entity, runner)) = with_children
        .get(trigger.entity())
        .ok()
        .and_then(|children| children.iter().find_map(|&child| runners.get(child).ok()))
    {
        if runner.is_line_finished() {
            commands.trigger_targets(UpdateRunner::NextLine, runner_entity);
        } 
    }
}

fn on_runner_updated_update_ui(
    trigger: Trigger<RunnerUpdated>,
    mut commands: Commands,
    progress: Res<State<Progress>>,
    text_box_wrapper: Single<Entity, With<TextBoxWrapper>>,
    instructions_text: Single<(Entity, &mut Text), With<Instructions>>,
    mut next_progress: ResMut<NextState<Progress>>,
) {
    let (instructions, mut instructions_text) = instructions_text.into_inner();
    match trigger.event() {
        RunnerUpdated::HideText => {
            commands
                .entity(text_box_wrapper.into_inner())
                .insert(Visibility::Hidden);
            commands.entity(instructions).insert(Visibility::Hidden);
        }
        RunnerUpdated::ShowText => {
            commands
                .entity(text_box_wrapper.into_inner())
                .insert(Visibility::Inherited);
        }
        RunnerUpdated::FinishedLine => {
            commands.entity(instructions).insert(Visibility::Inherited);
        }
        RunnerUpdated::ShowChoices => {
            instructions_text.0.clear();
            instructions_text.0.push_str("click an option to respond");
        }
        RunnerUpdated::HideChoices => {
            commands.entity(instructions).insert(Visibility::Hidden);
            instructions_text.0.clear();
            instructions_text.0.push_str("click to continue");
        }
        RunnerUpdated::FinishedMain => {
            debug_assert_eq!(progress.get(), &Progress::Show);
            next_progress.set(Progress::Postshow);
        }
        RunnerUpdated::FinishedEnd => {
            debug_assert_eq!(progress.get(), &Progress::Postshow);
            next_progress.set(Progress::Leaving);
        }
        _ => {}
    }
}

fn on_start_show(
    trigger: Trigger<RunnerUpdated>,
    mut commands: Commands,
    mut video_streams: ResMut<Assets<VideoStream>>,
    with_materials: Query<&MeshMaterial3d<StandardMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    future_video_players: Query<(Entity, &FutureVideoPlayer)>,
    audio_players: Query<&AudioSink, With<AudioPlayer<VideoStream>>>,
    progress: Res<State<Progress>>,
    mut next_progress: ResMut<NextState<Progress>>,
) {
    if *trigger.event() == RunnerUpdated::StartShow {
        for (_, video) in video_streams.iter_mut() {
            video.playing = true;
        }
        for (entity, player) in future_video_players.iter() {
            commands
                .entity(entity)
                .insert(VideoPlayer(player.0.clone()))
                .remove::<FutureVideoPlayer>();
            if let Ok(MeshMaterial3d(material_handle)) = with_materials.get(entity) {
                if let Some(material) = materials.get_mut(material_handle.id()) {
                    material.unlit = true;
                }
            }
        }
        for sink in audio_players.iter() {
            sink.play();
        }
        debug_assert_eq!(progress.get(), &Progress::Preshow);
        next_progress.set(Progress::Show);
    }
}

fn on_leave_animation_finished(
    _trigger: Trigger<AnimationFinished>,
    progress: Res<State<Progress>>,
    mut next_progress: ResMut<NextState<Progress>>,
) {
    // do something? fade to black?
    debug_assert_eq!(progress.get(), &Progress::Leaving);
    next_progress.set(Progress::End);
}

fn on_enter_animation_finished(
    _trigger: Trigger<AnimationFinished>,
    mut script_runners: Query<&mut ScriptRunner>,
    progress: Res<State<Progress>>,
    mut next_progress: ResMut<NextState<Progress>>,
) {
    for mut runner in script_runners.iter_mut() {
        runner.unpause();
    }
    debug_assert_eq!(progress.get(), &Progress::Entering);
    next_progress.set(Progress::Preshow);
}

fn on_video_finished(
    trigger: Trigger<VideoFinished>,
    mut commands: Commands,
    video_streams: Res<Assets<VideoStream>>,
    mut video_players: Query<(
        Entity,
        AnyOf<(&MeshMaterial3d<StandardMaterial>, &mut ScreenLight)>,
        &VideoPlayer,
    )>,
    audio_players: Query<(&AudioPlayer<VideoStream>, &AudioSink)>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    script_runners: Query<Entity, With<ScriptRunner>>,
    screen_off_image: Res<ScreenOffImage>,
    progress: Res<State<Progress>>,
    mut next_progress: ResMut<NextState<Progress>>,
) {
    info!("video finished!");
    let id = trigger.event().0;
    if video_streams.contains(id) {
        for entity in script_runners.iter() {
            commands.trigger_targets(UpdateRunner::ShowEnded, entity);
        }

        for (entity, (material, screen_light), player) in video_players.iter_mut() {
            if player.0.id() != id {
                continue;
            }
            // TODO: different image replacement?
            if let Some(handle) = material {
                if let Some(material) = materials.get_mut(handle.id()) {
                    material.base_color_texture = Some(screen_off_image.0.clone());
                    material.unlit = false;
                }
            } else if let Some(mut screen_light) = screen_light {
                screen_light.image = screen_off_image.0.clone();
            }
            commands.entity(entity).remove::<VideoPlayer>();
        }
        for (AudioPlayer(stream), sink) in audio_players.iter() {
            if stream.id() == id {
                sink.pause();
            }
        }
        debug_assert!(progress.get() == &Progress::Show || progress.get() == &Progress::Postshow);
        next_progress.set(Progress::Postshow);
    }
}

fn update_chair_materials(
    mut commands: Commands,
    roots: Query<Entity, With<Chair>>,
    with_children: Query<&Children>,
    not_updateds: Query<
        (Entity, &MeshMaterial3d<StandardMaterial>),
        Without<MeshMaterial3d<ExtendedMaterial<StandardMaterial, ScreenLightExtension>>>,
    >,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut extended_materials: ResMut<
        Assets<ExtendedMaterial<StandardMaterial, ScreenLightExtension>>,
    >,
    screen_light: Query<Entity, With<ScreenLight>>,
    mut to_load: ResMut<AssetsToLoad>,
    mut found_materials: Local<
        HashMap<
            AssetId<StandardMaterial>,
            Handle<ExtendedMaterial<StandardMaterial, ScreenLightExtension>>,
        >,
    >,
    mut updated_chairs: Local<usize>,
) {
    let Ok(screen_light) = screen_light.get_single() else {
        return;
    };
    for root in roots.iter() {
        for descendant in with_children.iter_descendants(root) {
            let Ok((entity, old_material_handle)) = not_updateds.get(descendant) else {
                continue;
            };
            if let Some(extended) = found_materials.get(&old_material_handle.id()) {
                commands
                    .entity(entity)
                    .insert(MeshMaterial3d(extended.clone()))
                    .remove::<MeshMaterial3d<StandardMaterial>>();
                *updated_chairs += 1;
                continue;
            }
            let Some(old_material) = materials.get_mut(old_material_handle.id()) else {
                continue;
            };
            let extended = extended_materials.add(ExtendedMaterial {
                base: old_material.clone(),
                extension: ScreenLightExtension {
                    light: screen_light,
                },
            });
            let extended_id = extended.id();
            found_materials.insert(old_material_handle.id(), extended.clone());
            commands
                .entity(entity)
                .insert(MeshMaterial3d(extended))
                .remove::<MeshMaterial3d<StandardMaterial>>();
            to_load.0.push(extended_id.untyped());
            *updated_chairs += 1;
        }
    }
    if *updated_chairs >= (CHAIR_ROWS * CHAIR_COLS) as usize {
        *updated_chairs = 0;
    }
}

fn check_loaded_state(
    assets: Res<AssetServer>,
    mut to_load: ResMut<AssetsToLoad>,
    loaded: Res<State<Loaded>>,
    mut next_loaded: ResMut<NextState<Loaded>>,
) {
    to_load.retain(|&id| assets.is_managed(id) && !assets.is_loaded(id));
    if to_load.is_empty() && !**loaded.get() {
        next_loaded.set(Loaded(true));
    }
}

fn remove_waiting_for_loads(
    mut commands: Commands,
    waiting_for_loads: Query<Entity, With<WaitingForLoads>>,
) {
    for waiting in waiting_for_loads.iter() {
        commands.entity(waiting).remove::<WaitingForLoads>();
    }
}

fn switch_to_theater_ui(
    mut commands: Commands,
    start_ui_root: Single<Entity, With<StartUiRoot>>,
    theater_root: Single<Entity, With<TheaterUiRoot>>,
) {
    // i am too lazy to do the Option<&mut Visibility> dance
    commands
        .entity(start_ui_root.into_inner())
        .despawn_recursive();
    commands
        .entity(theater_root.into_inner())
        .insert(Visibility::Inherited);
}

fn fade_in_to_theater(
    mut commands: Commands,
    start_ui_root: Single<Entity, With<StartUiRoot>>,
    mut animations: ResMut<Assets<AnimationClip>>,
    mut anim_graphs: ResMut<Assets<AnimationGraph>>,
) {
    let (target, player, graph) = alpha_fade_animation(
        1.,
        0.,
        4.,
        |color: &mut BackgroundColor| &mut color.0,
        "start_ui_root",
        &mut animations,
        |clip| clip.add_event(4., AnimationFinished),
    );
    let start_ui_root = start_ui_root.into_inner();
    commands
        .entity(start_ui_root)
        .insert((
            AnimationTarget {
                id: target,
                player: start_ui_root,
            },
            player,
            AnimationGraphHandle(anim_graphs.add(graph)),
        ))
        .observe(finish_fade_in_to_theater);
}

fn finish_fade_in_to_theater(
    _trigger: Trigger<AnimationFinished>,
    progress: Res<State<Progress>>,
    mut next_progress: ResMut<NextState<Progress>>,
) {
    debug_assert_eq!(progress.get(), &Progress::FadeIn);
    next_progress.set(Progress::Entering);
}

fn fade_to_black(
    mut commands: Commands,
    ui_root: Single<Entity, With<UiRoot>>,
    theater_root: Single<Entity, With<TheaterUiRoot>>,
    mut animations: ResMut<Assets<AnimationClip>>,
    mut anim_graphs: ResMut<Assets<AnimationGraph>>,
) {
    let end_ui = commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                align_self: AlignSelf::Stretch,
                justify_self: JustifySelf::Stretch,
                justify_content: JustifyContent::Center,
                justify_items: JustifyItems::Center,
                align_content: AlignContent::Center,
                align_items: AlignItems::Center,
                width: Val::Vw(100.),
                height: Val::Vh(100.),
                ..default()
            },
            BackgroundColor(Color::linear_rgba(0., 0., 0., 0.)),
        ))
        .id();
    let (end_ui_target, end_ui_anim_player, fade_graph) = alpha_fade_animation(
        0.,
        1.,
        4.,
        |bg: &mut BackgroundColor| &mut bg.0,
        "end_ui",
        &mut animations,
        |clip| clip.add_event(4., AnimationFinished),
    );
    commands
        .entity(end_ui)
        .insert((
            AnimationTarget {
                id: end_ui_target,
                player: end_ui,
            },
            end_ui_anim_player,
            AnimationGraphHandle(anim_graphs.add(fade_graph)),
        ))
        .set_parent(ui_root.into_inner())
        .observe(
            |trigger: Trigger<AnimationFinished>,
             mut commands: Commands,
             mut animations: ResMut<Assets<AnimationClip>>,
             mut anim_graphs: ResMut<Assets<AnimationGraph>>| {
                let end_text = commands
                    .spawn((
                        Node {
                            align_self: AlignSelf::Center,
                            justify_self: JustifySelf::Center,
                            align_content: AlignContent::Center,
                            justify_content: JustifyContent::Center,
                            align_items: AlignItems::Center,
                            justify_items: JustifyItems::Center,
                            ..default()
                        },
                        Text("the end\n(you can close the window now)".into()),
                        TextLayout {
                            justify: JustifyText::Center,
                            ..default()
                        },
                        TextFont {
                            font_size: 50.,
                            ..default()
                        },
                        TextColor(Color::linear_rgba(1., 1., 1., 0.)),
                    ))
                    .id();
                let (end_text_target, end_text_anim_player, fade_graph) = alpha_fade_animation(
                    0.,
                    1.,
                    2.,
                    |text_color: &mut TextColor| &mut text_color.0,
                    "end_text",
                    &mut animations,
                    |_| {},
                );
                commands.entity(end_text).insert((
                    AnimationTarget {
                        id: end_text_target,
                        player: end_text,
                    },
                    end_text_anim_player,
                    AnimationGraphHandle(anim_graphs.add(fade_graph)),
                ));
                commands.entity(trigger.entity()).add_child(end_text);
            },
        );
    commands
        .entity(theater_root.into_inner())
        .despawn_recursive();
}

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            MaterialPlugin::<ExtendedMaterial<StandardMaterial, Paper>>::default(),
            VideoPlugin,
            VideoPlayerPlugin::<MeshMaterial3d<StandardMaterial>>::default(),
            VideoPlayerPlugin::<ScreenLight>::default(),
            ScriptPlugin,
            ScreenLightPlugin,
        ))
        .init_state::<Progress>()
        .init_state::<Loaded>()
        .add_computed_state::<StatePair<Progress, Loaded>>()
        .add_computed_state::<StatePair<Loaded, Progress>>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                update_chair_materials,
                check_loaded_state.run_if(in_state(Loaded(false))),
            )
                .chain(),
        )
        .add_systems(
            OnEnter(Loaded(true)),
            (
                remove_waiting_for_loads,
                |mut commands: Commands, loading_text: Query<Entity, With<LoadingText>>| {
                    for entity in loading_text.iter() {
                        commands.entity(entity).despawn_recursive();
                    }
                },
            ),
        )
        .add_systems(
            OnEnter(Progress::Entering),
            (
                switch_to_theater_ui,
                |you: Single<&mut AnimationPlayer, With<You>>| {
                    you.into_inner().resume_all();
                },
            ),
        )
        .add_systems(
            OnEnter(Progress::Leaving),
            (|me: Single<&mut AnimationPlayer, With<Me>>| {
                me.into_inner().resume_all();
            },),
        )
        .add_systems(OnEnter(Progress::End), fade_to_black)
        .add_systems(
            OnEnter(StatePair(Progress::FadeIn, Loaded(true))),
            fade_in_to_theater,
        )
        .add_systems(Last, (debug_events::<RunnerUpdated>(),))
        .run();
}
