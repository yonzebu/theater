
use bevy::asset::UntypedAssetId;
use bevy::ecs::component;
use bevy::pbr::{ExtendedMaterial, MaterialExtension};
use bevy::render::render_resource::{AsBindGroup, ShaderRef};
use bevy::scene::SceneInstanceReady;
use bevy::{math::vec3, prelude::*};

mod debug;
use debug::*;
use video::{VideoPlugin, VideoStream};
mod video;
mod script;

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
struct VideoPlayer(Handle<VideoStream>);

#[derive(Component)]
#[component(storage = "SparseSet")]
struct Chair;

const CHAIR_ROWS: i32 = 5;
const CHAIR_COLS: i32 = 5;
const CHAIR_COUNT: i32 = 25;

#[derive(Resource)]
struct AssetsToLoad(Vec<UntypedAssetId>);

impl<I> From<I> for AssetsToLoad
where
    I: IntoIterator,
    I::Item: Into<UntypedAssetId>
{
    fn from(value: I) -> Self {
        AssetsToLoad(value.into_iter().map(|item| item.into()).collect())
    }
}

// #[derive(Component)]
struct WaitingForLoads;

impl Component for WaitingForLoads {
    const STORAGE_TYPE: component::StorageType = component::StorageType::Table;
    fn register_component_hooks(hooks: &mut component::ComponentHooks) {
        hooks.on_add(|mut deferred_world, entity, _| {
            if let Some(mut visibility) = deferred_world.get_mut::<Visibility>(entity) {
                *visibility = Visibility::Hidden;
            } else {
                deferred_world.commands().entity(entity).insert(Visibility::Hidden);
            }
        });
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut papers: ResMut<Assets<ExtendedMaterial<StandardMaterial, Paper>>>,
    assets: Res<AssetServer>,
) {

    commands.spawn((Camera3d::default(), Transform::from_xyz(0., 2., 5.)));
    commands.spawn((PointLight::default(), Transform::from_xyz(0., 2., -9.)));

    let rectangle = meshes.add(Rectangle::new(1.0, 1.0));
    let black = materials.add(Color::linear_rgb(0., 0., 0.));
    // screen
    commands.spawn((
        Mesh3d(rectangle.clone()),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::linear_rgb(0.5, 0.5, 0.5),
            unlit: true,
            ..default()
        })),
        Transform {
            translation: vec3(0., 2.5, -9.9),
            scale: vec3(18., 18. * 9. / 16., 1.),
            ..default()
        },
        VideoPlayer(assets.load::<VideoStream>("nonfinal/showtiny.mp4")),
        WaitingForLoads,
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
    ));

    // other walls
    let mut cube_mesh = Cuboid::from_size(vec3(20., 20., 15.)).mesh().build();
    cube_mesh.invert_winding().unwrap();
    commands.spawn((
        Mesh3d(meshes.add(cube_mesh)),
        MeshMaterial3d(materials.add(Color::linear_rgb(0.5, 0.125, 0.125))),
        Transform::from_xyz(0., 0., -5.),
        WaitingForLoads,
    ));

    let me_image = assets.load("nonfinal/maybem.png");
    let me_mesh = assets.load("nonfinal/me.glb#Mesh0/Primitive0");
    commands.spawn((
        Mesh3d(me_mesh.clone()),
        MeshMaterial3d(papers.add(
            ExtendedMaterial {
                base: StandardMaterial::from(me_image.clone()),
                extension: Paper {}
            }
        )),
        Transform::from_xyz(-2., 0., -0.5).looking_to(Dir3::Y, Dir3::Z).with_scale(Vec3::ONE * 0.5),
        WaitingForLoads,
    ));
    let you_image = assets.load("nonfinal/maybey.png");
    let you_mesh = assets.load("nonfinal/you.glb#Mesh0/Primitive0");
    commands.spawn((
        Mesh3d(you_mesh.clone()),
        MeshMaterial3d(papers.add(
            ExtendedMaterial {
                base: StandardMaterial::from(you_image.clone()),
                extension: Paper {}
            }
        )),
        Transform::from_xyz(2., 0., -0.5).looking_to(Dir3::Y, Dir3::Z).with_scale(Vec3::ONE * 0.5),
        WaitingForLoads
    
    ));

    // chairs
    let chair: Handle<Scene> = assets.load("nonfinal/chair2mat.glb#Scene0");
    for i in 0i32..CHAIR_COLS {
        for j in 0i32..CHAIR_ROWS {
            let x = 2. * (i - 2) as f32;
            let y = -j as f32 * 0.75;
            let z = -j as f32 * 2.;
            commands.spawn((
                // Mesh3d(meshes.add(Cuboid::default())),
                // Mesh3d(chair.clone()),
                // MeshMaterial3d(chair_material.clone()),
                SceneRoot(chair.clone()),
                Transform {
                    translation: vec3(x, y, z),
                    scale: Vec3::ONE * 0.25,
                    ..default()
                },
                DebugMarker,
                Chair,
                WaitingForLoads
            ));
        }
    }

    commands.insert_resource(AssetsToLoad(vec![
        me_image.id().untyped(),
        me_mesh.id().untyped(),
        you_image.id().untyped(),
        you_mesh.id().untyped(),
        chair.id().untyped()
    ]));
}

fn update_chair_materials(
    mut commands: Commands,
    roots: Query<Entity, With<Chair>>,
    with_children: Query<&Children>,
    not_updateds: Query<(Entity, &MeshMaterial3d<StandardMaterial>)>
) {
    for root in roots.iter() {
        for descendant in with_children.iter_descendants(root) {
            let Ok((entity, old_material)) = not_updateds.get(descendant) else {
                continue;
            };
            
        }
    }
}

fn update_video_player(
    video_player: Query<(&MeshMaterial3d<StandardMaterial>, &VideoPlayer)>,
    mut video_streams: ResMut<Assets<VideoStream>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    keyboard: Res<ButtonInput<KeyCode>>,
) {
    for (video_material, video_player) in video_player.iter() {
        if let (Some(material), Some(video_stream)) = (materials.get_mut(video_material.id()), video_streams.get_mut(video_player.0.id())) {
            material.base_color_texture = video_stream.buffered_frames.get(0).map(|frame| frame.image.clone());
            
            if keyboard.just_pressed(KeyCode::KeyV) {
                video_stream.playing = !video_stream.playing;
                println!("video playing = {}", video_stream.playing);
            }
        }
    }
}

fn check_loaded_state(
    mut commands: Commands,
    assets: Res<AssetServer>,
    to_load: Option<Res<AssetsToLoad>>,
    mut waiting_for_load: Query<(Entity, Option<&mut Visibility>), With<WaitingForLoads>>,
) {
    let Some(to_load) = to_load else {
        return;
    };
    if to_load.0.iter().all(|id| assets.is_loaded(*id)) {
        for (waiting, visibility) in waiting_for_load.iter_mut() {
            commands.entity(waiting).remove::<WaitingForLoads>();
            if let Some(mut visibility) = visibility {
                *visibility = Visibility::Inherited;
            }
        }
        commands.remove_resource::<AssetsToLoad>();
    }
}

fn update(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>
) {
}

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, MaterialPlugin::<ExtendedMaterial<StandardMaterial, Paper>>::default(), DebugPlugin, VideoPlugin))
        .add_systems(Startup, setup)
        .add_systems(Update, (
            check_loaded_state, 
            update_chair_materials, 
            update, 
            update_video_player
        ))
        // .add_systems(
        //     Last, 
        //     (
        //         debug_events::<AssetEvent<StandardMaterial>>(), 
        //         debug_events::<AssetEvent<Scene>>(),
        //     )
        // )
        .run();
}
