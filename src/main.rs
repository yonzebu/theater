#![allow(unused, clippy::type_complexity)]
use bevy::asset::UntypedAssetId;
use bevy::audio::PlaybackMode;
use bevy::ecs::component::ComponentId;
use bevy::ecs::world::DeferredWorld;
use bevy::pbr::{ExtendedMaterial, MaterialExtension, NotShadowCaster};
use bevy::render::render_resource::{AsBindGroup, Face, ShaderRef};
use bevy::utils::hashbrown::HashMap;
use bevy::{math::vec3, prelude::*};

mod debug;
use debug::*;
use screen_light::{
    ScreenLight, ScreenLightExtension, ScreenLightExtensionPlugin, ScreenLightPlugin,
};
use script::{RunnerUpdated, ScriptChoices, ScriptPlugin, ScriptRunner, UpdateRunner};
use video::{VideoPlayer, VideoPlayerPlugin, VideoPlugin, VideoStream, VideoStreamSettings};
mod screen_light;
mod script;
mod video;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, States)]
enum GameState {
    Loading(Progress),
    Active(Progress),
    // Paused(Progress)
}

impl Default for GameState {
    fn default() -> Self {
        GameState::Loading(Progress::Start)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, States)]
enum Progress {
    #[default]
    Start,
    Entering,
    Preshow,
    Show,
    EarlyLeave,
    Postshow,
    Leaving,
    End,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum LoadState {
    Loading,
    Loaded,
}

impl ComputedStates for LoadState {
    type SourceStates = GameState;
    fn compute(sources: Self::SourceStates) -> Option<Self> {
        match sources {
            GameState::Active(_) => Some(LoadState::Loaded),
            GameState::Loading(_) => Some(LoadState::Loading),
        }
    }
}

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

const CHAIR_ROWS: i32 = 5;
const CHAIR_COLS: i32 = 5;
const CHAIR_COUNT: i32 = 25;
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

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut papers: ResMut<Assets<ExtendedMaterial<StandardMaterial, Paper>>>,
    mut images: ResMut<Assets<Image>>,
    assets: Res<AssetServer>,
) {
    commands.spawn((Camera3d::default(), Transform::from_xyz(0., 2., 5.)));
    // commands.spawn((PointLight::default(), Transform::from_xyz(0., 0., -9.)));
    let video_stream = assets.load_with_settings(
        "nonfinal/testshow.mp4",
        |settings: &mut VideoStreamSettings| {
            settings.use_mips = false;
        },
    );
    // commands.spawn((
    //     AudioPlayer(video_stream.clone()),
    //     PlaybackSettings {
    //         mode: PlaybackMode::Once,
    //         paused: true,
    //         ..default()
    //     },
    // ));
    commands.spawn((
        // keep transform synced with screen transform
        Transform::from_xyz(0., 2.5, SCREEN_LIGHT_POS).looking_to(Dir3::Z, Dir3::Y),
        ScreenLight {
            image: Handle::default(),
        },
        VideoPlayer(video_stream.clone()),
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
            unlit: true,
            ..default()
        })),
        Transform {
            translation: vec3(0., 2.5, SCREEN_POS),
            scale: vec3(SCREEN_WIDTH, SCREEN_HEIGHT, 1.),
            ..default()
        },
        VideoPlayer(video_stream.clone()),
        WaitingForLoads,
        NotShadowCaster,
        Screen,
    ));
    // screen border
    commands.spawn((
        Mesh3d(rectangle.clone()),
        MeshMaterial3d(black.clone()),
        Transform {
            translation: vec3(0., 2.5, -9.9),
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

    let me_image = assets.load("me.png");
    let me_mesh = assets.load("me.glb#Mesh0/Primitive0");
    commands.spawn((
        Mesh3d(me_mesh.clone()),
        MeshMaterial3d(papers.add(ExtendedMaterial {
            base: StandardMaterial::from(me_image.clone()),
            extension: Paper {},
        })),
        Transform::from_xyz(-2., 0., -0.5)
            .looking_to(Dir3::Y, Dir3::Z)
            .with_scale(Vec3::ONE * 0.5),
        WaitingForLoads,
    ));
    let you_image = assets.load("you.png");
    let you_mesh = assets.load("you.glb#Mesh0/Primitive0");
    commands.spawn((
        Mesh3d(you_mesh.clone()),
        MeshMaterial3d(papers.add(ExtendedMaterial {
            base: StandardMaterial::from(you_image.clone()),
            extension: Paper {},
        })),
        Transform::from_xyz(2., 0., -0.5)
            .looking_to(Dir3::Y, Dir3::Z)
            .with_scale(Vec3::ONE * 0.5),
        WaitingForLoads,
    ));

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

    let script = assets.load("nonfinal/testscript.txt");
    let script_choices_entity = commands.spawn_empty().id();
    let mut script_runner = ScriptRunner::new(script.clone(), script_choices_entity, 10.0);
    // script_runner.pause();
    let root = commands
        .spawn((
            Node {
                flex_direction: FlexDirection::ColumnReverse,
                align_self: AlignSelf::Stretch,
                justify_self: JustifySelf::Stretch,
                flex_wrap: FlexWrap::NoWrap,
                justify_content: JustifyContent::FlexStart,
                align_items: AlignItems::End,
                align_content: AlignContent::Center,
                padding: UiRect::horizontal(Val::VMin(30.)).with_bottom(Val::VMin(1.)),
                ..default()
            },
            Visibility::Inherited,
        ))
        .id();
    let text_box_wrapper = commands
        .spawn((Node {
            align_self: AlignSelf::Stretch,
            height: Val::Percent(20.),
            margin: UiRect::horizontal(Val::VMin(5.)),
            ..default()
        },))
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
    commands.entity(root).add_child(text_box_wrapper);
    commands
        .entity(text_box_wrapper)
        .add_child(text_visible_box);
    commands.entity(text_visible_box).add_child(script_runner);

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
    commands.entity(root).add_child(choice_box_wrapper);
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
        } else {
            commands.trigger_targets(UpdateRunner::FinishLine, runner_entity);
        }
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
        }
    }
}

fn update_video_player(
    audio_players: Query<(&AudioPlayer<VideoStream>, &AudioSink)>,
    mut video_streams: ResMut<Assets<VideoStream>>,
    keyboard: Res<ButtonInput<KeyCode>>,
) {
    if keyboard.just_pressed(KeyCode::KeyV) {
        for (_, video_stream) in video_streams.iter_mut() {
            video_stream.playing = !video_stream.playing;
            println!("video playing = {}", video_stream.playing);
        }
    }
    for (audio_player, audio_sink) in audio_players.iter() {
        if let Some(video_stream) = video_streams.get(audio_player.0.id()) {
            if video_stream.playing {
                audio_sink.play();
            } else {
                audio_sink.pause();
            }
        }
    }
}

fn check_loaded_state(
    assets: Res<AssetServer>,
    mut to_load: ResMut<AssetsToLoad>,
    game_state: Res<State<GameState>>,
    mut next_game_state: ResMut<NextState<GameState>>,
) {
    to_load.retain(|&id| !assets.is_loaded(id));
    if to_load.len() == 0 {
        match game_state.get() {
            &GameState::Loading(progress) => next_game_state.set(GameState::Active(progress)),
            _ => {}
        }
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

fn update(mut runners: Query<&mut ScriptRunner>, keyboard: Res<ButtonInput<KeyCode>>) {
    if !keyboard.just_pressed(KeyCode::KeyT) {
        return;
    }
    for mut runner in runners.iter_mut() {
        if runner.paused() {
            runner.unpause();
        } else {
            runner.pause();
        }
    }
}

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            MaterialPlugin::<ExtendedMaterial<StandardMaterial, Paper>>::default(),
            DebugPlugin,
            VideoPlugin,
            VideoPlayerPlugin::<MeshMaterial3d<StandardMaterial>>::default(),
            VideoPlayerPlugin::<ScreenLight>::default(),
            ScriptPlugin,
            ScreenLightPlugin,
        ))
        .init_state::<GameState>()
        .add_computed_state::<LoadState>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                (
                    update_chair_materials,
                    check_loaded_state.run_if(in_state(LoadState::Loading)),
                )
                    .chain(),
                update,
                update_video_player,
            ),
        )
        .add_systems(
            OnTransition {
                exited: LoadState::Loading,
                entered: LoadState::Loaded,
            },
            remove_waiting_for_loads,
        )
        .add_systems(Last, (debug_events::<RunnerUpdated>(),))
        .run();
}
