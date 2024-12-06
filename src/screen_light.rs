//! This is held together by glue and toothpicks.
//!
//! It's not really intended to be used outside of this project, and has a lot of problems that were
//! solved with the most convenient thing at the time and never got changed to something "better"
//! because it was never needed. There are several rendering features that will probably break this
//! that haven't been tested at all (e.g. deferred, lightmaps, custom materials)

use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::DerefMut;
use std::sync::Arc;

use bevy::ecs::entity::EntityHashMap;
use bevy::ecs::system::lifetimeless::{Read, SQuery, SRes};
use bevy::ecs::system::SystemParamItem;
use bevy::pbr::{
    prepare_lights, queue_material_meshes, queue_shadows, DrawPrepass, ExtendedMaterial,
    LightViewEntities, MaterialPipeline, MaterialPipelineKey, MeshPipelineKey, NotShadowCaster,
    PreparedMaterial, PrepassPipeline, PrepassPipelinePlugin, PrepassPlugin, RenderLightmaps,
    RenderMaterialInstances, RenderMeshInstanceFlags, RenderMeshInstances,
    RenderVisibleMeshEntities, Shadow, ShadowBinKey, ShadowView, SimulationLightSystems,
    ViewLightEntities, VisibleMeshEntities,
};
use bevy::render::camera::CameraProjection;
use bevy::render::mesh::RenderMesh;
use bevy::render::primitives::{Aabb, Frustum};
use bevy::render::render_asset::{prepare_assets, RenderAssetPlugin, RenderAssets};
use bevy::render::render_phase::{
    AddRenderCommand, BinnedRenderPhaseType, DrawFunctions, ViewBinnedRenderPhases,
};
use bevy::render::render_resource::binding_types::{
    sampler, texture_2d, texture_depth_2d, uniform_buffer,
};
use bevy::render::render_resource::{
    encase, AsBindGroup, AsBindGroupError, BindGroupLayout, BindGroupLayoutEntries,
    BindGroupLayoutEntry, BufferInitDescriptor, BufferUsages, Extent3d, OwnedBindingResource,
    PipelineCache, SamplerBindingType, ShaderStages, ShaderType, SpecializedMeshPipelines,
    TextureAspect, TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType,
    TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension, UnpreparedBindGroup,
};
use bevy::render::renderer::RenderDevice;
use bevy::render::sync_component::SyncComponentPlugin;
use bevy::render::sync_world::{MainEntity, RenderEntity, TemporaryRenderEntity};
use bevy::render::texture::{CachedTexture, DepthAttachment, GpuImage, TextureCache};
use bevy::render::view::{
    check_visibility, ExtractedView, NoFrustumCulling, RenderLayers, VisibilityRange,
    VisibilitySystems, VisibleEntityRanges,
};
use bevy::render::{Extract, Render, RenderApp, RenderSet};
use bevy::utils::{HashMap, Parallel};
use bevy::{pbr::MaterialExtension, prelude::*};

use crate::video::ReceiveFrame;

// TODO: make this a Resource?
const SCREEN_LIGHT_SHADOW_MAP_WIDTH: u32 = 1920;

#[derive(Default, ShaderType, Clone)]
pub struct ScreenLightUniform {
    clip_from_world: Mat4,
    image_size: UVec2,
    forward_dir: Vec3,
    light_pos: Vec3,
}

#[derive(Component)]
#[require(Projection, Frustum, VisibleMeshEntities, Visibility, Transform)]
pub struct ScreenLight {
    pub image: Handle<Image>,
}

impl ReceiveFrame for ScreenLight {
    type Param = ();

    fn should_receive(
        &self,
        frame: &Handle<Image>,
        _: &<Self::Param as bevy::ecs::system::SystemParam>::Item<'_, '_>,
    ) -> bool {
        self.image.id() != frame.id()
    }

    fn receive_frame(
        &mut self,
        frame: Handle<Image>,
        _: &mut <Self::Param as bevy::ecs::system::SystemParam>::Item<'_, '_>,
    ) {
        self.image = frame;
    }
}

#[derive(Component)]
struct ScreenLightEntity {
    light_entity: Entity,
}

#[derive(Component)]
pub struct ExtractedScreenLight {
    image: Handle<Image>,
    uniform: ScreenLightUniform,
    clip_from_view: Mat4,
    transform: GlobalTransform,
}

#[derive(Default, Deref, DerefMut, Resource)]
pub struct ExtractedScreenLights(EntityHashMap<RenderEntity>);

#[derive(Resource)]
pub struct ViewScreenLightShadowTexture {
    _texture: CachedTexture,
    texture_view: TextureView,
}

/// Intended for use with StandardMaterial (and extensions). Uses bindings 96-99.
#[derive(Clone, Asset, TypePath)]
pub struct ScreenLightExtension {
    pub light: Entity,
}

impl AsBindGroup for ScreenLightExtension {
    type Data = ();
    type Param = (
        SRes<ExtractedScreenLights>,
        SRes<RenderAssets<GpuImage>>,
        Option<SRes<ViewScreenLightShadowTexture>>,
        SQuery<Read<ExtractedScreenLight>>,
    );

    fn label() -> Option<&'static str> {
        Some("screen light extension")
    }

    fn unprepared_bind_group(
        &self,
        _layout: &BindGroupLayout,
        render_device: &RenderDevice,
        (screen_lights_map, images, shadow_texture, screen_lights): &mut SystemParamItem<
            '_,
            '_,
            Self::Param,
        >,
    ) -> Result<UnpreparedBindGroup<Self::Data>, AsBindGroupError> {
        let shadow_texture = shadow_texture
            .as_ref()
            .ok_or(AsBindGroupError::RetryNextUpdate)?;
        let screen_light = screen_lights_map
            .get(&self.light)
            .and_then(|light| screen_lights.get(light.id()).ok())
            .ok_or(AsBindGroupError::RetryNextUpdate)?;
        let image = images
            .get(screen_light.image.id())
            .ok_or(AsBindGroupError::RetryNextUpdate)?;

        let mut size_buf = encase::UniformBuffer::new(Vec::new());
        size_buf.write(&screen_light.uniform).unwrap();
        let uniform = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: "screen light uniform".into(),
            contents: size_buf.as_ref(),
            usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM,
        });
        Ok(UnpreparedBindGroup {
            bindings: vec![
                (96, OwnedBindingResource::Buffer(uniform)),
                (
                    97,
                    OwnedBindingResource::TextureView(shadow_texture.texture_view.clone()),
                ),
                (
                    98,
                    OwnedBindingResource::TextureView(image.texture_view.clone()),
                ),
                (99, OwnedBindingResource::Sampler(image.sampler.clone())),
            ],
            data: (),
        })
    }

    fn bind_group_layout_entries(_render_device: &RenderDevice) -> Vec<BindGroupLayoutEntry>
    where
        Self: Sized,
    {
        BindGroupLayoutEntries::with_indices(
            ShaderStages::FRAGMENT,
            (
                (96, uniform_buffer::<ScreenLightUniform>(false)),
                (97, texture_depth_2d()),
                (
                    98,
                    texture_2d(TextureSampleType::Float { filterable: true }),
                ),
                (99, sampler(SamplerBindingType::Filtering)),
            ),
        )
        .to_vec()
    }
}

impl MaterialExtension for ScreenLightExtension {
    fn fragment_shader() -> bevy::render::render_resource::ShaderRef {
        "screen_light.wgsl".into()
    }
}

type Extended<M> = ExtendedMaterial<M, ScreenLightExtension>;

// this should not be pub bc certain things become harder to reason about
#[derive(Clone, Asset, AsBindGroup, TypePath)]
struct ScreenLightPrepassExtension {}

impl MaterialExtension for ScreenLightPrepassExtension {}

type PrepassExt<M> = ExtendedMaterial<M, ScreenLightPrepassExtension>;

/// I don't think this works with deferred rendering, I haven't tested and don't care right now.
pub struct ScreenLightPlugin;

impl Plugin for ScreenLightPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            SyncComponentPlugin::<ScreenLight>::default(),
            ScreenLightExtensionPlugin::<StandardMaterial>::default(),
        ))
        .add_systems(
            PostUpdate,
            (
                update_screen_light_frusta
                    .in_set(SimulationLightSystems::UpdateLightFrusta)
                    .after(TransformSystem::TransformPropagate)
                    .after(SimulationLightSystems::AssignLightsToClusters),
                check_visibility::<With<ScreenLight>>.in_set(VisibilitySystems::CheckVisibility),
                check_screen_light_mesh_visibility
                    .in_set(SimulationLightSystems::CheckLightVisibility)
                    .after(VisibilitySystems::CalculateBounds)
                    .after(TransformSystem::TransformPropagate)
                    .after(SimulationLightSystems::UpdateLightFrusta)
                    .after(VisibilitySystems::CheckVisibility),
            ),
        );

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<ExtractedScreenLights>()
            .add_systems(ExtractSchedule, extract_screen_lights)
            .add_systems(
                Render,
                (prepare_screen_lights
                    .in_set(RenderSet::ManageViews)
                    .after(prepare_lights)
                    .after(prepare_assets::<GpuImage>),),
            );

        render_app.world_mut().add_observer(
            |trigger: Trigger<OnAdd, ExtractedScreenLight>, mut commands: Commands| {
                if let Some(mut entity) = commands.get_entity(trigger.entity()) {
                    entity.insert(LightViewEntities::default());
                }
            },
        );
        render_app.world_mut().add_observer(
            |trigger: Trigger<OnRemove, ExtractedScreenLight>, mut commands: Commands| {
                if let Some(mut entity) = commands.get_entity(trigger.entity()) {
                    entity.remove::<LightViewEntities>();
                }
            },
        );
        render_app.world_mut().add_observer(
            |trigger: Trigger<OnRemove, ExtractedScreenLight>,
             query: Query<&LightViewEntities>,
             mut commands: Commands| {
                if let Ok(light_view_entities) = query.get(trigger.entity()) {
                    for light_views in light_view_entities.values() {
                        for &light_view in light_views.iter() {
                            if let Some(mut light_view) = commands.get_entity(light_view) {
                                light_view.despawn();
                            }
                        }
                    }
                }
            },
        );
    }
}

/// What it says on the tin. Mostly duplicated from bevy's MaterialPlugin, but with shadows always on and
/// without adding any other passes.
struct MaterialPluginOnlyShadow<M: Material> {
    prepass_enabled: bool,
    _marker: PhantomData<M>,
}

impl<M: Material> Default for MaterialPluginOnlyShadow<M> {
    fn default() -> Self {
        MaterialPluginOnlyShadow {
            prepass_enabled: true,
            _marker: PhantomData,
        }
    }
}

impl<M: Material> Plugin for MaterialPluginOnlyShadow<M>
where
    M::Data: Clone + Eq + Hash,
{
    fn build(&self, app: &mut App) {
        app.init_asset::<M>()
            .register_type::<MeshMaterial3d<M>>()
            .add_plugins(RenderAssetPlugin::<PreparedMaterial<M>>::default());

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<DrawFunctions<Shadow>>()
                .init_resource::<RenderMaterialInstances<M>>()
                .add_render_command::<Shadow, DrawPrepass<M>>()
                .init_resource::<SpecializedMeshPipelines<MaterialPipeline<M>>>()
                .add_systems(ExtractSchedule, extract_mesh_materials::<M>);

            render_app.add_systems(
                Render,
                queue_shadows::<M>
                    .in_set(RenderSet::QueueMeshes)
                    .after(prepare_assets::<PreparedMaterial<M>>),
            );
        }

        app.add_plugins(PrepassPipelinePlugin::<M>::default());

        if self.prepass_enabled {
            app.add_plugins(PrepassPlugin::<M>::default());
        }
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.init_resource::<MaterialPipeline<M>>();
        }
    }
}

/// Adds the necessary stuff for using [`ScreenLightExtension`] with a given material. Note that
/// you must also add [`ScreenLightPlugin`], and that [`ScreenLightPlugin`] also adds an instance
/// of this plugin for [`StandardMaterial`], so you don't need to manually add it.
pub struct ScreenLightExtensionPlugin<M: Material> {
    // pub prepass_enabled: bool,
    pub _marker: PhantomData<M>,
}

impl<M: Material> Default for ScreenLightExtensionPlugin<M> {
    fn default() -> Self {
        ScreenLightExtensionPlugin {
            // prepass_enabled: true,
            _marker: PhantomData,
        }
    }
}

impl<M: Material> Plugin for ScreenLightExtensionPlugin<M>
where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    fn build(&self, app: &mut App) {
        app.init_resource::<ExtendedToPrepass<M>>()
            .add_systems(
                PostUpdate,
                (
                    update_screen_light_materials::<M>.after(update_screen_light_frusta),
                    update_screen_light_prepass_extensions::<M>,
                ),
            )
            .add_plugins((
                MaterialPlugin::<Extended<M>> {
                    shadows_enabled: false,
                    prepass_enabled: false,
                    _marker: PhantomData,
                },
                MaterialPluginOnlyShadow::<PrepassExt<M>> {
                    prepass_enabled: false,
                    _marker: PhantomData,
                },
            ))
            .add_observer(on_remove_screen_light_extension::<M>);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.add_systems(
            Render,
            (queue_screen_light_shadows::<PrepassExt<M>>
                .in_set(RenderSet::QueueMeshes)
                .after(prepare_assets::<PreparedMaterial<PrepassExt<M>>>),),
        );
    }
}

#[derive(Resource, Deref, DerefMut)]
struct ExtendedToPrepass<M: Material>(HashMap<AssetId<Extended<M>>, AssetId<PrepassExt<M>>>);

impl<M: Material> Default for ExtendedToPrepass<M> {
    fn default() -> Self {
        ExtendedToPrepass(HashMap::default())
    }
}

#[derive(Component)]
struct WithPrepassExt<M: Material> {
    old_material: Handle<Extended<M>>,
}

fn update_screen_light_prepass_extensions<M: Material>(
    mut commands: Commands,
    without_prepass: Query<
        (Entity, &MeshMaterial3d<Extended<M>>),
        Without<MeshMaterial3d<PrepassExt<M>>>,
    >,
    mut changed: Query<
        (
            &MeshMaterial3d<Extended<M>>,
            &mut MeshMaterial3d<PrepassExt<M>>,
            &mut WithPrepassExt<M>,
        ),
        Changed<MeshMaterial3d<Extended<M>>>,
    >,
    materials: Res<Assets<Extended<M>>>,
    mut prepass_materials: ResMut<Assets<PrepassExt<M>>>,
    mut extended_to_prepass: ResMut<ExtendedToPrepass<M>>,
) {
    for (entity, material) in without_prepass.iter() {
        if let Some(&prepass_id) = extended_to_prepass.get(&material.id()) {
            if let Some(prepass_handle) = prepass_materials.get_strong_handle(prepass_id) {
                commands.entity(entity).insert((
                    MeshMaterial3d(prepass_handle),
                    WithPrepassExt {
                        old_material: material.0.clone(),
                    },
                ));
                continue;
            } else {
                // clearly it's not a valid id anymore, so we just create a new prepass extension
                extended_to_prepass.remove(&material.id());
            }
        }
        if let Some(extended) = materials.get(material.id()) {
            let prepass_handle = prepass_materials.add(ExtendedMaterial {
                base: extended.base.clone(),
                extension: ScreenLightPrepassExtension {},
            });
            let prepass_id = prepass_handle.id();
            commands.entity(entity).insert((
                MeshMaterial3d(prepass_handle),
                WithPrepassExt {
                    old_material: material.0.clone(),
                },
            ));
            extended_to_prepass.insert(material.id(), prepass_id);
        }
    }

    for (MeshMaterial3d(new_handle), mut prepass_ext, mut old_material) in changed.iter_mut() {
        let old_handle = std::mem::replace(&mut old_material.old_material, new_handle.clone());
        let old_id = old_handle.id();
        match old_handle {
            Handle::Strong(strong) => {
                if Arc::into_inner(strong).is_some() {
                    extended_to_prepass.remove(&old_id);
                }
            }
            Handle::Weak(id) => {
                if !materials.contains(id) {
                    extended_to_prepass.remove(&id);
                }
            }
        }
        if let Some(&prepass_id) = extended_to_prepass.get(&new_handle.id()) {
            if let Some(prepass_handle) = prepass_materials.get_strong_handle(prepass_id) {
                prepass_ext.0 = prepass_handle;
                continue;
            } else {
                extended_to_prepass.remove(&new_handle.id());
            }
        }
        if let Some(new_material) = materials.get(new_handle.id()) {
            let prepass_handle = prepass_materials.add(ExtendedMaterial {
                base: new_material.base.clone(),
                extension: ScreenLightPrepassExtension {},
            });
            prepass_ext.0 = prepass_handle;
        }
    }
}

fn on_remove_screen_light_extension<M: Material>(
    trigger: Trigger<OnRemove, MeshMaterial3d<Extended<M>>>,
    mut commands: Commands,
    mut with_prepass_exts: Query<&mut WithPrepassExt<M>>,
    mut extended_to_prepass: ResMut<ExtendedToPrepass<M>>,
    materials: Res<Assets<Extended<M>>>,
) {
    if let Some(mut entity_commands) = commands.get_entity(trigger.entity()) {
        entity_commands.remove::<(MeshMaterial3d<PrepassExt<M>>, WithPrepassExt<M>)>();
    }
    if let Ok(mut with_prepass_ext) = with_prepass_exts.get_mut(trigger.entity()) {
        let weak = with_prepass_ext.old_material.clone_weak();
        let old_material = std::mem::replace(&mut with_prepass_ext.old_material, weak);
        let old_id = old_material.id();
        match old_material {
            Handle::Strong(strong) => {
                if Arc::into_inner(strong).is_some() {
                    extended_to_prepass.remove(&old_id);
                }
            }
            Handle::Weak(id) => {
                if !materials.contains(id) {
                    extended_to_prepass.remove(&id);
                }
            }
        }
    }
}

fn extract_mesh_materials<M: Material>(
    mut material_instances: ResMut<RenderMaterialInstances<M>>,
    query: Extract<Query<(Entity, &ViewVisibility, &MeshMaterial3d<M>)>>,
) {
    material_instances.clear();

    for (entity, view_visibility, material) in &query {
        if view_visibility.get() {
            material_instances.insert(entity.into(), material.id());
        }
    }
}

fn update_screen_light_materials<M: Material>(
    screen_lights: Query<
        Entity,
        (
            With<ScreenLight>,
            Or<(Added<ScreenLight>, Added<Frustum>, Changed<ScreenLight>, Changed<Frustum>)>,
        ),
    >,
    materials: Res<Assets<Extended<M>>>,
    mut asset_events: EventWriter<AssetEvent<Extended<M>>>,
) {
    for (id, material) in materials.iter() {
        if screen_lights.contains(material.extension.light) {
            asset_events.send(AssetEvent::Modified { id });
        }
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn update_screen_light_frusta(
    mut views: Query<
        (&GlobalTransform, &Projection, &mut Frustum),
        (
            Or<(Changed<GlobalTransform>, Changed<Projection>)>,
            With<ScreenLight>,
        ),
    >,
) {
    for (transform, projection, mut frustum) in &mut views {
        *frustum = projection.compute_frustum(transform);
    }
}

fn shrink_entities(visible_entities: &mut Vec<Entity>) {
    // Check that visible entities capacity() is no more than two times greater than len()
    let capacity = visible_entities.capacity();
    let reserved = capacity
        .checked_div(visible_entities.len())
        .map_or(0, |reserve| {
            if reserve > 2 {
                capacity / (reserve / 2)
            } else {
                capacity
            }
        });

    visible_entities.shrink_to(reserved);
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn check_screen_light_mesh_visibility(
    mut screen_lights: Query<
        (&Frustum, &mut VisibleMeshEntities, Option<&RenderLayers>),
        With<ScreenLight>,
    >,
    mut visible_entity_query: Query<
        (
            Entity,
            &InheritedVisibility,
            &mut ViewVisibility,
            Option<&RenderLayers>,
            Option<&Aabb>,
            Option<&GlobalTransform>,
            Has<VisibilityRange>,
            Has<NoFrustumCulling>,
        ),
        (
            Without<NotShadowCaster>,
            Without<DirectionalLight>,
            With<Mesh3d>,
        ),
    >,
    visible_entity_ranges: Option<Res<VisibleEntityRanges>>,
    mut spot_visible_entities_queue: Local<Parallel<Vec<Entity>>>,
) {
    let visible_entity_ranges = visible_entity_ranges.as_deref();
    for (frustum, mut visible_entities, maybe_view_mask) in screen_lights.iter_mut() {
        visible_entities.clear();

        let view_mask = maybe_view_mask.unwrap_or_default();

        visible_entity_query.par_iter_mut().for_each_init(
            || spot_visible_entities_queue.borrow_local_mut(),
            |spot_visible_entities_local_queue,
             (
                entity,
                inherited_visibility,
                mut view_visibility,
                maybe_entity_mask,
                maybe_aabb,
                maybe_transform,
                has_visibility_range,
                has_no_frustum_culling,
            )| {
                if !inherited_visibility.get() {
                    return;
                }

                let entity_mask = maybe_entity_mask.unwrap_or_default();
                if !view_mask.intersects(entity_mask) {
                    return;
                }
                // Check visibility ranges.
                if has_visibility_range
                    && visible_entity_ranges.is_some_and(|visible_entity_ranges| {
                        !visible_entity_ranges.entity_is_in_range_of_any_view(entity)
                    })
                {
                    return;
                }

                if let (Some(aabb), Some(transform)) = (maybe_aabb, maybe_transform) {
                    let model_to_world = transform.affine();

                    if has_no_frustum_culling
                        || frustum.intersects_obb(aabb, &model_to_world, true, true)
                    {
                        view_visibility.set();
                        spot_visible_entities_local_queue.push(entity);
                    }
                } else {
                    view_visibility.set();
                    spot_visible_entities_local_queue.push(entity);
                }
            },
        );

        for entities in spot_visible_entities_queue.iter_mut() {
            visible_entities.append(entities);
        }

        shrink_entities(visible_entities.deref_mut());
    }
}

fn create_render_visible_mesh_entities(
    commands: &mut Commands,
    mapper: &Extract<Query<RenderEntity>>,
    visible_entities: &VisibleMeshEntities,
) -> RenderVisibleMeshEntities {
    RenderVisibleMeshEntities {
        entities: visible_entities
            .iter()
            .map(|e| {
                let render_entity = mapper
                    .get(*e)
                    .unwrap_or_else(|_| commands.spawn(TemporaryRenderEntity).id());
                (render_entity, MainEntity::from(*e))
            })
            .collect(),
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn extract_screen_lights(
    mut commands: Commands,
    screen_lights: Extract<
        Query<(
            Entity,
            RenderEntity,
            &ScreenLight,
            &VisibleMeshEntities,
            &GlobalTransform,
            &ViewVisibility,
            &Frustum,
            &Projection,
        )>,
    >,
    images: Extract<Res<Assets<Image>>>,
    mapper: Extract<Query<RenderEntity>>,
    mut extracted_lights: ResMut<ExtractedScreenLights>,
    // i am scared to mess with this param bc it might break something but it can probably be removed
    mut previous_lights_len: Local<usize>,
) {
    extracted_lights.clear();
    let mut lights_to_spawn = Vec::with_capacity(*previous_lights_len);
    for (
        main_entity,
        render_entity,
        screen_light,
        visible_entities,
        transform,
        _view_visibility,
        frustum,
        projection,
    ) in screen_lights.iter()
    {
        // if !view_visibility.get() {
        //     continue;
        // }
        let Some(image) = images.get(screen_light.image.id()) else {
            continue;
        };
        extracted_lights.insert(main_entity, render_entity.into());

        let render_visible_entities =
            create_render_visible_mesh_entities(&mut commands, &mapper, visible_entities);

        let clip_from_view = projection.get_clip_from_view();
        let world_from_view = transform.compute_matrix();
        let clip_from_world = clip_from_view * world_from_view.inverse();

        lights_to_spawn.push((
            render_entity,
            (
                ExtractedScreenLight {
                    image: screen_light.image.clone(),
                    uniform: ScreenLightUniform {
                        clip_from_world,
                        image_size: image.size(),
                        forward_dir: *transform.forward(),
                        light_pos: transform.translation(),
                    },
                    clip_from_view,
                    transform: *transform,
                },
                render_visible_entities,
                *frustum,
            ),
        ));
    }
    *previous_lights_len = lights_to_spawn.len();
    commands.insert_or_spawn_batch(lights_to_spawn);
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn prepare_screen_lights(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    mut views: Query<(Entity, &mut ViewLightEntities), (With<Camera3d>, With<ExtractedView>)>,
    mut shadow_render_phases: ResMut<ViewBinnedRenderPhases<Shadow>>,
    screen_lights: Query<(Entity, &ExtractedScreenLight, &Frustum)>,
    mut light_view_entities: Query<&mut LightViewEntities>,
    images: Res<RenderAssets<GpuImage>>,
) {
    // set up light data for each view
    for (entity, mut view_light_entities) in views.iter_mut() {
        let mut view_lights = Vec::new();

        let Ok((light_entity, screen_light, frustum)) = screen_lights.get_single() else {
            continue;
        };

        let Ok(mut light_view_entities) = light_view_entities.get_mut(light_entity) else {
            continue;
        };

        let Some(image) = images.get(screen_light.image.id()) else {
            continue;
        };
        let shadow_map = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: "screen light shadow map".into(),
                size: Extent3d {
                    width: SCREEN_LIGHT_SHADOW_MAP_WIDTH,
                    height: (SCREEN_LIGHT_SHADOW_MAP_WIDTH * image.texture.width()) / image.texture.height(),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Depth32Float,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            },
        );
        let shadow_map_view = shadow_map.texture.create_view(&TextureViewDescriptor {
            label: "screen light shadow map view".into(),
            format: Some(TextureFormat::Depth32Float),
            dimension: Some(TextureViewDimension::D2),
            aspect: TextureAspect::DepthOnly,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });

        let light_view_entities = light_view_entities
            .entry(entity)
            .or_insert_with(|| vec![commands.spawn_empty().id()]);
        let view_light_entity = light_view_entities[0];

        commands.entity(view_light_entity).insert((
            ShadowView {
                depth_attachment: DepthAttachment::new(shadow_map_view.clone(), Some(0.0)),
                pass_name: "shadow pass screen light".into(),
            },
            ExtractedView {
                viewport: UVec4::new(
                    0,
                    0,
                    screen_light.uniform.image_size.x,
                    screen_light.uniform.image_size.y,
                ),
                world_from_view: screen_light.transform,
                clip_from_view: screen_light.clip_from_view,
                clip_from_world: Some(screen_light.uniform.clip_from_world),
                hdr: false,
                color_grading: Default::default(),
            },
            *frustum,
            ScreenLightEntity { light_entity },
        ));

        view_lights.push(view_light_entity);

        shadow_render_phases.insert_or_clear(view_light_entity);

        if !view_light_entities.lights.contains(&view_light_entity) {
            view_light_entities.lights.push(view_light_entity);
        }
        commands.insert_resource(ViewScreenLightShadowTexture {
            _texture: shadow_map,
            texture_view: shadow_map_view,
        });
    }

    // FIXME: this might be important to keep but i'm worried about it breaking something
    // // Despawn light-view entities for views that no longer exist
    // for mut entities in &mut light_view_entities {
    //     for (_, light_view_entities) in
    //         entities.extract_if(|entity, _| !live_views.contains(entity))
    //     {
    //         despawn_entities(&mut commands, light_view_entities);
    //     }
    // }
}

// this can't rely on [`queue_shadows`] bc that checks for kind of light and this is a custom light it's not looking for
#[allow(clippy::too_many_arguments)]
fn queue_screen_light_shadows<M: Material>(
    shadow_draw_functions: Res<DrawFunctions<Shadow>>,
    prepass_pipeline: Res<PrepassPipeline<M>>,
    render_meshes: Res<RenderAssets<RenderMesh>>,
    render_mesh_instances: Res<RenderMeshInstances>,
    render_materials: Res<RenderAssets<PreparedMaterial<M>>>,
    render_material_instances: Res<RenderMaterialInstances<M>>,
    mut shadow_render_phases: ResMut<ViewBinnedRenderPhases<Shadow>>,
    mut pipelines: ResMut<SpecializedMeshPipelines<PrepassPipeline<M>>>,
    pipeline_cache: Res<PipelineCache>,
    view_lights: Query<(Entity, &ViewLightEntities)>,
    screen_light_entities: Query<&ScreenLightEntity>,
    screen_light_entities_visibility: Query<&RenderVisibleMeshEntities, With<ExtractedScreenLight>>,
) where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    for (_view_entity, view_lights) in &view_lights {
        let draw_shadow_mesh = shadow_draw_functions.read().id::<DrawPrepass<M>>();
        for view_light_entity in view_lights.lights.iter().copied() {
            let Ok(light_entity) = screen_light_entities.get(view_light_entity) else {
                continue;
            };
            let shadow_phase = shadow_render_phases.get_mut(&view_light_entity).unwrap();

            let visible_entities = screen_light_entities_visibility
                .get(light_entity.light_entity)
                .unwrap();

            let light_key = MeshPipelineKey::DEPTH_PREPASS;

            for (entity, main_entity) in visible_entities.iter().copied() {
                let Some(mesh_instance) = render_mesh_instances.render_mesh_queue_data(main_entity)
                else {
                    continue;
                };
                if !mesh_instance
                    .flags
                    .contains(RenderMeshInstanceFlags::SHADOW_CASTER)
                {
                    continue;
                }
                let Some(material_asset_id) = render_material_instances.get(&main_entity) else {
                    continue;
                };
                let Some(material) = render_materials.get(*material_asset_id) else {
                    continue;
                };
                let Some(mesh) = render_meshes.get(mesh_instance.mesh_asset_id) else {
                    continue;
                };

                let mut mesh_key =
                    light_key | MeshPipelineKey::from_bits_retain(mesh.key_bits.bits());

                mesh_key |= match material.properties.alpha_mode {
                    AlphaMode::Mask(_)
                    | AlphaMode::Blend
                    | AlphaMode::Premultiplied
                    | AlphaMode::Add
                    | AlphaMode::AlphaToCoverage => MeshPipelineKey::MAY_DISCARD,
                    _ => MeshPipelineKey::NONE,
                };
                let pipeline_id = pipelines.specialize(
                    &pipeline_cache,
                    &prepass_pipeline,
                    MaterialPipelineKey {
                        mesh_key,
                        bind_group_data: material.key.clone(),
                    },
                    &mesh.layout,
                );

                let pipeline_id = match pipeline_id {
                    Ok(id) => id,
                    Err(err) => {
                        error!("{}", err);
                        continue;
                    }
                };

                mesh_instance
                    .material_bind_group_id
                    .set(material.get_bind_group_id());

                shadow_phase.add(
                    ShadowBinKey {
                        draw_function: draw_shadow_mesh,
                        pipeline: pipeline_id,
                        asset_id: mesh_instance.mesh_asset_id.into(),
                    },
                    (entity, main_entity),
                    BinnedRenderPhaseType::mesh(mesh_instance.should_batch()),
                );
            }
        }
    }
}
