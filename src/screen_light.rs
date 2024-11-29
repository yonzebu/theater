//! this is held together by glue and toothpicks

use std::marker::PhantomData;
use std::ops::DerefMut;
use std::hash::Hash;

use bevy::ecs::entity::EntityHashMap;
use bevy::ecs::system::lifetimeless::{Read, SQuery, SRes};
use bevy::ecs::system::SystemParamItem;
use bevy::pbr::{prepare_lights, DrawPrepass, ExtendedMaterial, LightViewEntities, MaterialPipelineKey, MeshPipelineKey, NotShadowCaster, PreparedMaterial, PrepassPipeline, RenderMaterialInstances, RenderMeshInstanceFlags, RenderMeshInstances, RenderVisibleMeshEntities, Shadow, ShadowBinKey, ShadowView, SimulationLightSystems, ViewLightEntities, VisibleClusterableObjects, VisibleMeshEntities};
use bevy::render::camera::CameraProjection;
use bevy::render::mesh::RenderMesh;
use bevy::render::primitives::{Aabb, Frustum};
use bevy::render::render_asset::{prepare_assets, RenderAssets};
use bevy::render::render_phase::{BinnedRenderPhaseType, DrawFunctions, ViewBinnedRenderPhases};
use bevy::render::render_resource::binding_types::{sampler, texture_2d, texture_depth_2d, uniform_buffer};
use bevy::render::renderer::RenderDevice;
use bevy::render::sync_component::SyncComponentPlugin;
use bevy::render::sync_world::{MainEntity, RenderEntity, TemporaryRenderEntity};
use bevy::render::texture::{CachedTexture, DepthAttachment, GpuImage, TextureCache};
use bevy::render::view::{check_visibility, ExtractedView, NoFrustumCulling, RenderLayers, VisibilityRange, VisibilitySystems, VisibleEntityRanges};
use bevy::render::{Extract, Render, RenderApp, RenderSet};
use bevy::utils::Parallel;
use bevy::{pbr::MaterialExtension, pbr::update_spot_light_frusta, prelude::*};
use bevy::render::render_resource::{encase, AsBindGroup, AsBindGroupError, BindGroupLayout, BindGroupLayoutEntries, BindGroupLayoutEntry, Buffer, BufferInitDescriptor, BufferUsages, CachedRenderPipelineId, Extent3d, OwnedBindingResource, PipelineCache, SamplerBindingType, ShaderStages, ShaderType, SpecializedMeshPipelines, TextureAspect, TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension, UnpreparedBindGroup};

use crate::video::ReceiveFrame;

#[derive(Default, ShaderType, Clone)]
pub struct ScreenLightUniform {
    image_size: UVec4,
    clip_from_world: Mat4,
}

#[derive(Component)]
#[require(Projection, Frustum, VisibleMeshEntities, Visibility, Transform)]
pub struct ScreenLight {
    pub image: Handle<Image>,
}

impl ReceiveFrame for ScreenLight {
    type SystemParam = ();
    fn should_receive(&self, id: impl Into<AssetId<Image>>) -> bool {
        self.image.id() != id.into()
    }
    fn receive_frame(
        &mut self,
        frame: Handle<Image>,
        _: &mut <Self::SystemParam as bevy::ecs::system::SystemParam>::Item<'_, '_>,
    ) {
        self.image = frame;
    }
}

#[derive(Component)]
struct ScreenLightEntity {
    light_entity: Entity
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
    texture: CachedTexture,
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
        (screen_lights_map, images, shadow_texture, screen_lights): &mut SystemParamItem<'_, '_, Self::Param>,
    ) -> Result<UnpreparedBindGroup<Self::Data>, AsBindGroupError> {
        let shadow_texture = shadow_texture.as_ref().ok_or(AsBindGroupError::RetryNextUpdate)?;
        let screen_light = screen_lights_map
            .get(&self.light)
            .and_then(|light| screen_lights.get(light.id()).ok())
            .ok_or(AsBindGroupError::RetryNextUpdate)?;
        
        let image = images.get(screen_light.image.id()).ok_or(AsBindGroupError::RetryNextUpdate)?;
        
        let mut size_buf = encase::UniformBuffer::new(Vec::new());
        size_buf.write(&screen_light.uniform).unwrap();
        let uniform = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: "screen light uniform".into(),
            contents: size_buf.as_ref(),
            usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM
        });
        Ok(UnpreparedBindGroup {
            bindings: vec![
                (96, OwnedBindingResource::Buffer(uniform)),
                (97, OwnedBindingResource::TextureView(shadow_texture.texture_view.clone())),
                (98, OwnedBindingResource::TextureView(image.texture_view.clone())),
                (99, OwnedBindingResource::Sampler(image.sampler.clone()))
            ],
            data: ()
        })
    }
    fn bind_group_layout_entries(_render_device: &RenderDevice) -> Vec<BindGroupLayoutEntry>
    where
        Self: Sized 
    {
        BindGroupLayoutEntries::with_indices(
            ShaderStages::FRAGMENT, 
            (
                (96, uniform_buffer::<ScreenLightUniform>(false)),
                (97, texture_depth_2d()),
                (98, texture_2d(TextureSampleType::Float { filterable: true })),
                (99, sampler(SamplerBindingType::Filtering))
            )
        ).to_vec()
    }
}

impl MaterialExtension for ScreenLightExtension {
    fn fragment_shader() -> bevy::render::render_resource::ShaderRef {
        "screen_light.wgsl".into()
    }
}

pub struct ScreenLightPlugin;

impl Plugin for ScreenLightPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_plugins(SyncComponentPlugin::<ScreenLight>::default())
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
                        // NOTE: This MUST be scheduled AFTER the core renderer visibility check
                        // because that resets entity `ViewVisibility` for the first view
                        // which would override any results from this otherwise
                        .after(VisibilitySystems::CheckVisibility),
                )
            );
        
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.init_resource::<ExtractedScreenLights>()
            .add_systems(ExtractSchedule, extract_screen_lights)
            .add_systems(
                Render,
                (
                    prepare_screen_lights
                        .in_set(RenderSet::ManageViews)
                        .after(prepare_lights)
                        .after(prepare_assets::<GpuImage>),
                ),
            );

            render_app.world_mut().add_observer(|trigger: Trigger<OnAdd, ExtractedScreenLight>, mut commands: Commands| {    
                if let Some(mut entity) = commands.get_entity(trigger.entity()) {
                    entity.insert(LightViewEntities::default());
                }
            });
            render_app.world_mut().add_observer(|trigger: Trigger<OnRemove, ExtractedScreenLight>, mut commands: Commands| {
                if let Some(mut entity) = commands.get_entity(trigger.entity()) {
                    entity.remove::<LightViewEntities>();
                }
            });
            render_app.world_mut().add_observer(|trigger: Trigger<OnRemove, ExtractedScreenLight>, query: Query<&LightViewEntities>, mut commands: Commands| {
                if let Ok(light_view_entities) = query.get(trigger.entity()) {
                    for light_views in light_view_entities.values() {
                        for &light_view in light_views.iter() {
                            if let Some(mut light_view) = commands.get_entity(light_view) {
                                light_view.despawn();
                            }
                        }
                    }
                }
            });
    }
}

#[derive(Default)]
pub struct ScreenLightExtensionPlugin<M: Material>(PhantomData<M>);

impl<M: Material> Plugin for ScreenLightExtensionPlugin<M>
where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    fn build(&self, app: &mut App) {
        app.add_systems(PostUpdate, update_screen_light_materials::<M>.after(update_screen_light_frusta));
        
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_systems(
                Render,
                (
                    queue_screen_light_shadows::<M>
                        .in_set(RenderSet::QueueMeshes)
                        .after(prepare_assets::<PreparedMaterial<M>>),
                    // queue_screen_light_shadows::<ExtendedMaterial<M, ScreenLightExtension>>
                    //     .in_set(RenderSet::QueueMeshes)
                    //     .after(prepare_assets::<PreparedMaterial<ExtendedMaterial<M, ScreenLightExtension>>>),
                ),
            );
    }
}

fn update_screen_light_materials<M: Material>(
    screen_lights: Query<Entity, (With<ScreenLight>, Changed<ScreenLight>)>,
    assets: Res<Assets<ExtendedMaterial<M, ScreenLightExtension>>>,
    mut asset_events: EventWriter<AssetEvent<ExtendedMaterial<M, ScreenLightExtension>>>
) {
    for (id, material) in assets.iter() {
        if screen_lights.contains(material.extension.light) {
            asset_events.send(AssetEvent::Modified { id });
        }
    }
}

/// copied from [`update_spot_light_frusta`] with several modifications
fn update_screen_light_frusta(
    mut views: Query<
        (&GlobalTransform, &Projection, &mut Frustum),
        (Or<(Changed<GlobalTransform>, Changed<Projection>)>, With<ScreenLight>),
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

fn check_screen_light_mesh_visibility(
    mut screen_lights: Query<(
        &ScreenLight,
        &GlobalTransform,
        &Frustum,
        &mut VisibleMeshEntities,
        Option<&RenderLayers>,
    )>,
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
    for (
        screen_light, 
        transform, 
        frustum, 
        mut visible_entities, 
        maybe_view_mask
    ) in screen_lights.iter_mut() {
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
        view_visibility,
        frustum,
        projection,
    ) in screen_lights.iter() {
        // if !view_visibility.get() {
        //     continue;
        // }
        let Some(image) = images.get(screen_light.image.id()) else {
            continue;
        };
        extracted_lights.insert(main_entity, render_entity.into());

        // let view_backward = transform.back();
        // let near = match projection {
        //     Projection::Orthographic(ortho) => ortho.near,
        //     Projection::Perspective(perspective) => perspective.near,
        // };
        // let translate = GlobalTransform::from(Transform::from_translation(view_backward * near));

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
                        image_size: UVec4::from((image.size(), 0, 0)),
                        clip_from_world,
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


#[allow(clippy::too_many_arguments)]
pub fn prepare_screen_lights(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    mut views: Query<
        (
            Entity,
            &mut ViewLightEntities,
        ),
        (With<Camera3d>, With<ExtractedView>),
    >,
    mut shadow_render_phases: ResMut<ViewBinnedRenderPhases<Shadow>>,
    screen_lights: Query<(
        Entity,
        &ExtractedScreenLight,
        &Frustum,
    )>,
    mut light_view_entities: Query<&mut LightViewEntities>,
    images: Res<RenderAssets<GpuImage>>
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
                    width: image.texture.width(),
                    height: image.texture.height(),
                    depth_or_array_layers: 1,
                }, 
                mip_level_count: 1, 
                sample_count: 1, 
                dimension: TextureDimension::D2, 
                format: TextureFormat::Depth32Float, 
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT, 
                view_formats: &[], 
            }
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
                    screen_light.uniform.image_size.y
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
            texture: shadow_map,
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


// this can't rely on [`queue_shadows`] bc this checks for kind of light
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
    for (entity, view_lights) in &view_lights {
        let draw_shadow_mesh = shadow_draw_functions.read().id::<DrawPrepass<M>>();
        for view_light_entity in view_lights.lights.iter().copied() {
            let Ok(light_entity) = screen_light_entities.get(view_light_entity) else {
                continue;
            };
            let shadow_phase = shadow_render_phases.get_mut(&view_light_entity).unwrap();

            let visible_entities = screen_light_entities_visibility.get(light_entity.light_entity).unwrap();
            
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

                // // Even though we don't use the lightmap in the shadow map, the
                // // `SetMeshBindGroup` render command will bind the data for it. So
                // // we need to include the appropriate flag in the mesh pipeline key
                // // to ensure that the necessary bind group layout entries are
                // // present.
                // if render_lightmaps.render_lightmaps.contains_key(&main_entity) {
                //     mesh_key |= MeshPipelineKey::LIGHTMAPPED;
                // }

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
