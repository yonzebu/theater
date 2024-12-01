#![allow(unused)]
use bevy::{
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    math::vec3,
    prelude::*,
};
use std::fmt::{Debug, Write};

#[derive(Component, Reflect)]
pub struct DebugMarker;

fn debug_freecam(
    mut main_camera: Query<&mut Transform, With<Camera3d>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
) {
    let dt = time.delta_secs();
    let w = keyboard.pressed(KeyCode::KeyW) as i32;
    let a = keyboard.pressed(KeyCode::KeyA) as i32;
    let s = keyboard.pressed(KeyCode::KeyS) as i32;
    let d = keyboard.pressed(KeyCode::KeyD) as i32;
    let space = keyboard.pressed(KeyCode::Space) as i32;
    let shift = keyboard.pressed(KeyCode::ShiftLeft) as i32;
    let x = (d - a) as f32;
    let y = (space - shift) as f32;
    let z = (s - w) as f32;
    for mut cam_transform in main_camera.iter_mut() {
        cam_transform.translation += vec3(x, y, z) * 10. * dt;

        if keyboard.just_pressed(KeyCode::ArrowLeft) {
            cam_transform.look_to(Dir3::NEG_X, Dir3::Y);
        }
        if keyboard.just_pressed(KeyCode::ArrowRight) {
            cam_transform.look_to(Dir3::X, Dir3::Y);
        }
        if keyboard.just_pressed(KeyCode::ArrowUp) {
            cam_transform.look_to(Dir3::NEG_Z, Dir3::Y);
        }
        if keyboard.just_pressed(KeyCode::ArrowDown) {
            cam_transform.look_to(Dir3::Z, Dir3::Y);
        }
    }
}

fn debug(world: &mut World) {
    if !world
        .resource::<ButtonInput<KeyCode>>()
        .just_pressed(KeyCode::KeyP)
    {
        return;
    }

    let mut camera_query = world.query::<(&Camera, &Transform)>();
    if let Ok((_, transform)) = camera_query.get_single(world) {
        println!("camera transform: {transform:?}");
    };
    let mut marked_query = world.query::<(&DebugMarker, EntityRef)>();
    for (_, marked_data) in marked_query.iter(world) {
        // for id in marked_data.archetype().components() {
        //     println!("found component {:?}", world.components().get_info(id).map(|info| info.name()))
        // }
        // let Mesh3d(mesh_handle) = marked_data.get::<Mesh3d>().unwrap();
        // let meshes = world.resource::<Assets<Mesh>>();
        // println!("mesh handle: {mesh_handle:?}");
        // let mesh = meshes.get(mesh_handle).unwrap();
        // let aabb = mesh.compute_aabb().unwrap();
        // println!("aabb: {aabb:?}");
        // println!("marked transform: {marked_data:?}");
    }
}

#[derive(Component)]
#[component(storage = "SparseSet")]
struct FpsDisplay;

fn spawn_fps_display(mut commands: Commands) {
    commands.spawn((Text::new("fps: "), FpsDisplay));
}

fn update_fps_display(
    mut fps_displays: Query<&mut Text, With<FpsDisplay>>,
    diagnostics: Res<DiagnosticsStore>,
) {
    let diagnostic = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS).unwrap();
    let fps = diagnostic
        .measurement()
        .map(|measurement| measurement.value)
        .unwrap_or(0.);
    let avg_fps = diagnostic.average().unwrap_or(0.);
    for mut fps_display in fps_displays.iter_mut() {
        fps_display.0.clear();
        write!(&mut fps_display.0, "fps: {fps}\navg fps: {avg_fps}").unwrap();
    }
}

pub fn debug_events<E: Event + Debug>() -> impl FnMut(EventReader<E>) {
    |mut events| {
        for event in events.read() {
            println!("received event: {event:?}");
        }
    }
}

pub struct DebugPlugin;

impl Plugin for DebugPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FrameTimeDiagnosticsPlugin)
            .add_systems(Startup, spawn_fps_display)
            .add_systems(Update, (debug_freecam, update_fps_display))
            .add_systems(Last, debug);
    }
}
