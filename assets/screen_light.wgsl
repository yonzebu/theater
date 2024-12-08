#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    prepass_io::{Vertex, VertexOutput, FragmentOutput},
    pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
    forward_io::{Vertex, VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}
#endif // PREPASS_PIPELINE
#import bevy_pbr::{
    utils::{rand_f, interleaved_gradient_noise},
    utils,
    lighting,
    lighting::{LAYER_BASE, LAYER_CLEARCOAT, LightingInput},
    mesh_bindings::mesh,
    mesh_functions,
    mesh_view_bindings::globals,
    mesh_view_bindings as view_bindings,
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
    pbr_functions,
    pbr_types,
    view_transformations::position_world_to_clip
}
#import bevy_render::maths::PI

fn sample_shadow_map_hardware(
    depth_texture: texture_depth_2d, 
    shadow_sampler: sampler_comparison,
    light_local: vec2<f32>, 
    depth: f32
) -> f32 {
    return textureSampleCompare(
        depth_texture,
        shadow_sampler,
        light_local,
        depth,
    );
}

fn sample_shadow_map_castano_thirteen(
    shadow_map: texture_depth_2d,
    shadow_sampler: sampler_comparison,
    light_local: vec2<f32>, 
    depth: f32
) -> f32 {
    let shadow_map_size = vec2<f32>(textureDimensions(shadow_map));
    let inv_shadow_map_size = 1.0 / shadow_map_size;

    let uv = light_local * shadow_map_size;
    var base_uv = floor(uv + 0.5);
    let s = (uv.x + 0.5 - base_uv.x);
    let t = (uv.y + 0.5 - base_uv.y);
    base_uv -= 0.5;
    base_uv *= inv_shadow_map_size;

    let uw0 = (4.0 - 3.0 * s);
    let uw1 = 7.0;
    let uw2 = (1.0 + 3.0 * s);

    let u0 = (3.0 - 2.0 * s) / uw0 - 2.0;
    let u1 = (3.0 + s) / uw1;
    let u2 = s / uw2 + 2.0;

    let vw0 = (4.0 - 3.0 * t);
    let vw1 = 7.0;
    let vw2 = (1.0 + 3.0 * t);

    let v0 = (3.0 - 2.0 * t) / vw0 - 2.0;
    let v1 = (3.0 + t) / vw1;
    let v2 = t / vw2 + 2.0;

    var sum = 0.0;

    sum += uw0 * vw0 * sample_shadow_map_hardware(shadow_map, shadow_sampler, base_uv + (vec2(u0, v0) * inv_shadow_map_size), depth);
    sum += uw1 * vw0 * sample_shadow_map_hardware(shadow_map, shadow_sampler, base_uv + (vec2(u1, v0) * inv_shadow_map_size), depth);
    sum += uw2 * vw0 * sample_shadow_map_hardware(shadow_map, shadow_sampler, base_uv + (vec2(u2, v0) * inv_shadow_map_size), depth);

    sum += uw0 * vw1 * sample_shadow_map_hardware(shadow_map, shadow_sampler, base_uv + (vec2(u0, v1) * inv_shadow_map_size), depth);
    sum += uw1 * vw1 * sample_shadow_map_hardware(shadow_map, shadow_sampler, base_uv + (vec2(u1, v1) * inv_shadow_map_size), depth);
    sum += uw2 * vw1 * sample_shadow_map_hardware(shadow_map, shadow_sampler, base_uv + (vec2(u2, v1) * inv_shadow_map_size), depth);

    sum += uw0 * vw2 * sample_shadow_map_hardware(shadow_map, shadow_sampler, base_uv + (vec2(u0, v2) * inv_shadow_map_size), depth);
    sum += uw1 * vw2 * sample_shadow_map_hardware(shadow_map, shadow_sampler, base_uv + (vec2(u1, v2) * inv_shadow_map_size), depth);
    sum += uw2 * vw2 * sample_shadow_map_hardware(shadow_map, shadow_sampler, base_uv + (vec2(u2, v2) * inv_shadow_map_size), depth);

    return sum * (1.0 / 144.0);
}

fn map(min1: f32, max1: f32, min2: f32, max2: f32, value: f32) -> f32 {
    return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
}

fn random_rotation_matrix(scale: vec2<f32>, temporal: bool) -> mat2x2<f32> {
    let random_angle = 2.0 * PI * interleaved_gradient_noise(
        scale, select(1u, view_bindings::globals.frame_count, temporal));
    let m = vec2(sin(random_angle), cos(random_angle));
    return mat2x2(
        m.y, -m.x,
        m.x, m.y
    );
}

fn calculate_uv_offset_scale_jimenez_fourteen(texel_size: f32, blur_size: f32) -> vec2<f32> {
    let shadow_map_size = vec2<f32>(textureDimensions(view_bindings::directional_shadow_textures));

    // Empirically chosen fudge factor to make PCF look better across different CSM cascades
    let f = map(0.00390625, 0.022949219, 0.015, 0.035, texel_size);
    return f * blur_size / (texel_size * shadow_map_size);
}

fn sample_shadow_map_jimenez_fourteen(
    shadow_map: texture_depth_2d,
    shadow_sampler: sampler_comparison,
    light_local: vec2<f32>,
    depth: f32,
    texel_size: f32,
    blur_size: f32,
    temporal: bool,
) -> f32 {
    let shadow_map_size = vec2<f32>(textureDimensions(shadow_map));
    let rotation_matrix = random_rotation_matrix(light_local * shadow_map_size, temporal);
    let uv_offset_scale = calculate_uv_offset_scale_jimenez_fourteen(texel_size, blur_size);

    // https://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare (slides 120-135)
    let sample_offset0 = (rotation_matrix * utils::SPIRAL_OFFSET_0_) * uv_offset_scale;
    let sample_offset1 = (rotation_matrix * utils::SPIRAL_OFFSET_1_) * uv_offset_scale;
    let sample_offset2 = (rotation_matrix * utils::SPIRAL_OFFSET_2_) * uv_offset_scale;
    let sample_offset3 = (rotation_matrix * utils::SPIRAL_OFFSET_3_) * uv_offset_scale;
    let sample_offset4 = (rotation_matrix * utils::SPIRAL_OFFSET_4_) * uv_offset_scale;
    let sample_offset5 = (rotation_matrix * utils::SPIRAL_OFFSET_5_) * uv_offset_scale;
    let sample_offset6 = (rotation_matrix * utils::SPIRAL_OFFSET_6_) * uv_offset_scale;
    let sample_offset7 = (rotation_matrix * utils::SPIRAL_OFFSET_7_) * uv_offset_scale;

    var sum = 0.0;
    sum += sample_shadow_map_hardware(shadow_map, shadow_sampler, light_local + sample_offset0, depth);
    sum += sample_shadow_map_hardware(shadow_map, shadow_sampler, light_local + sample_offset1, depth);
    sum += sample_shadow_map_hardware(shadow_map, shadow_sampler, light_local + sample_offset2, depth);
    sum += sample_shadow_map_hardware(shadow_map, shadow_sampler, light_local + sample_offset3, depth);
    sum += sample_shadow_map_hardware(shadow_map, shadow_sampler, light_local + sample_offset4, depth);
    sum += sample_shadow_map_hardware(shadow_map, shadow_sampler, light_local + sample_offset5, depth);
    sum += sample_shadow_map_hardware(shadow_map, shadow_sampler, light_local + sample_offset6, depth);
    sum += sample_shadow_map_hardware(shadow_map, shadow_sampler, light_local + sample_offset7, depth);
    return sum / 8.0;
}

fn sample_shadow_map(
    shadow_map: texture_depth_2d,
    shadow_sampler: sampler_comparison,
    light_uv: vec2<f32>, 
    depth: f32, 
    texel_size: f32
) -> f32 {
    return sample_shadow_map_castano_thirteen(shadow_map, shadow_sampler, light_uv, depth);
//     return sample_shadow_map_jimenez_fourteen(shadow_map, shadow_sampler, light_uv, depth, texel_size, 1.0, false);
//     return sample_shadow_map_hardware(shadow_map, shadow_sampler, light_uv, depth);
}

fn sample_screen(
    screen: texture_2d<f32>,
    screen_sampler: sampler,
    light_local: vec2<f32>
) -> vec4<f32> {
    let screen_size = vec2<f32>(textureDimensions(screen));
    let inv_screen_size = 1.0 / screen_size;
    var sum = vec4(0.0);
    for (var x: i32 = 0; x < 5; x += 1) {
        for (var y: i32 = 0; y < 5; y += 1) {
            let offset = (vec2(f32(x), f32(y)) - vec2(2.0, 2.0)) * 5.0 * inv_screen_size;
            let offset_uv = light_local + offset;
            sum += textureSample(screen, screen_sampler, offset_uv);
        }
    }
    return sum / 25.0;
    // let uv = light_local * screen_size;
    // var base_uv = floor(uv + 0.5);
    // let s = (uv.x + 0.5 - base_uv.x);
    // let t = (uv.y + 0.5 - base_uv.y);
    // base_uv -= 0.5;
    // base_uv *= inv_screen_size;

    // let uw0 = (4.0 - 3.0 * s);
    // let uw1 = 7.0;
    // let uw2 = (1.0 + 3.0 * s);

    // let u0 = (3.0 - 2.0 * s) / uw0 - 2.0;
    // let u1 = (3.0 + s) / uw1;
    // let u2 = s / uw2 + 2.0;

    // let vw0 = (4.0 - 3.0 * t);
    // let vw1 = 7.0;
    // let vw2 = (1.0 + 3.0 * t);

    // let v0 = (3.0 - 2.0 * t) / vw0 - 2.0;
    // let v1 = (3.0 + t) / vw1;
    // let v2 = t / vw2 + 2.0;

    // var sum = vec4(0.0);

    // sum += uw0 * vw0 * textureSample(screen, screen_sampler, base_uv + (vec2(u0, v0) * inv_screen_size));
    // sum += uw1 * vw0 * textureSample(screen, screen_sampler, base_uv + (vec2(u1, v0) * inv_screen_size));
    // sum += uw2 * vw0 * textureSample(screen, screen_sampler, base_uv + (vec2(u2, v0) * inv_screen_size));

    // sum += uw0 * vw1 * textureSample(screen, screen_sampler, base_uv + (vec2(u0, v1) * inv_screen_size));
    // sum += uw1 * vw1 * textureSample(screen, screen_sampler, base_uv + (vec2(u1, v1) * inv_screen_size));
    // sum += uw2 * vw1 * textureSample(screen, screen_sampler, base_uv + (vec2(u2, v1) * inv_screen_size));

    // sum += uw0 * vw2 * textureSample(screen, screen_sampler, base_uv + (vec2(u0, v2) * inv_screen_size));
    // sum += uw1 * vw2 * textureSample(screen, screen_sampler, base_uv + (vec2(u1, v2) * inv_screen_size));
    // sum += uw2 * vw2 * textureSample(screen, screen_sampler, base_uv + (vec2(u2, v2) * inv_screen_size));

    // return sum * (1.0 / 144.0);
}

const APPROX_TEXEL_SIZE: f32 = 0.002;
// these are taken from bevy's default biases for spot lights
const DEPTH_BIAS: f32 = 0.02;
// ...except bevy also multiplies the normal bias by some values, so these are precalculated based on approximate expected shadow map size (1920x1080)
// ...and then futzed with using a fudge factor a little bit
const NORMAL_BIAS: f32 = 1.8 * 1.4142135 * APPROX_TEXEL_SIZE * 4.0;

fn fetch_screen_light_color(
    light_pos: vec3<f32>, 
    light_dir: vec3<f32>,
    world_position: vec3<f32>, 
    surface_normal: vec3<f32>,
    shadow_map: texture_depth_2d,
    shadow_sampler: sampler_comparison,
) -> vec3<f32> {
    let light_to_surface = world_position - light_pos;
    let distance_to_light = dot(light_dir, light_to_surface);

    let offset_position =
        world_position
        + (DEPTH_BIAS * normalize(light_to_surface))
        + (surface_normal * NORMAL_BIAS) * distance_to_light;
    let clip_pos = screen_light.clip_from_world * vec4(offset_position, 1.0);
    let ndc_pos = clip_pos.xyz / clip_pos.w;
    let shadow_uv = ndc_pos.xy * vec2(0.5, -0.5) + vec2(0.5);

    let shadow = sample_shadow_map(shadow_map, shadow_sampler, shadow_uv, ndc_pos.z, APPROX_TEXEL_SIZE);
    let sampled = sample_screen(screen_image, screen_sampler, shadow_uv);
    let color = sampled.rgb * clamp(sampled.a, 0., 1.);
    return shadow * color * 0.25;
}

fn screen_light_contrib(
    light_pos: vec3<f32>, 
    light_color: vec3<f32>, 
    input: ptr<function, LightingInput>
) -> vec3<f32> {
    // Unpack.
    let diffuse_color = (*input).diffuse_color;
    let P = (*input).P;
    let N = (*input).layers[LAYER_BASE].N;
    let V = (*input).V;

    let light_to_frag = light_pos - P;
    let L = normalize(light_to_frag);

    // Base layer
    
    // point lights do a thing with changing the specular intensity? but i'm not sure if i need that if i'm not doing range attenuation
    var specular_derived_input = lighting::derive_lighting_input(N, V, L);

#ifdef STANDARD_MATERIAL_ANISOTROPY
    let specular_light = lighting::specular_anisotropy(input, &specular_derived_input, L, 1.0);
#else   // STANDARD_MATERIAL_ANISOTROPY
    let specular_light = lighting::specular(input, &specular_derived_input, 1.0);
#endif  // STANDARD_MATERIAL_ANISOTROPY

    // Clearcoat

#ifdef STANDARD_MATERIAL_CLEARCOAT
    // Unpack.
    let clearcoat_N = (*input).layers[LAYER_CLEARCOAT].N;
    let clearcoat_strength = (*input).clearcoat_strength;

    // Perform specular input calculations again for the clearcoat layer. We
    // can't reuse the above because the clearcoat normal might be different
    // from the main layer normal.=
    var clearcoat_specular_derived_input =
        derive_lighting_input(clearcoat_N, V, L);

    // Calculate the specular light.
    let Fc_Frc = specular_clearcoat(
        input,
        &clearcoat_specular_derived_input,
        clearcoat_strength,
        1.0
    );
    let inv_Fc = 1.0 - Fc_Frc.r;    // Inverse Fresnel term.
    let Frc = Fc_Frc.g;             // Clearcoat light.
#endif  // STANDARD_MATERIAL_CLEARCOAT

    // Diffuse.
    // Comes after specular since its N⋅L is used in the lighting equation.
    var derived_input = lighting::derive_lighting_input(N, V, L);
    let diffuse = diffuse_color * lighting::Fd_Burley(input, &derived_input);

    // See https://google.github.io/filament/Filament.html#mjx-eqn-pointLightLuminanceEquation
    // Lout = f(v,l) Φ / { 4 π d^2 }⟨n⋅l⟩
    // where
    // f(v,l) = (f_d(v,l) + f_r(v,l)) * light_color
    // Φ is luminous power in lumens
    // our rangeAttenuation = 1 / d^2 multiplied with an attenuation factor for smoothing at the edge of the non-physical maximum light radius

    // For a point light, luminous intensity, I, in lumens per steradian is given by:
    // I = Φ / 4 π
    // The derivation of this can be seen here: https://google.github.io/filament/Filament.html#mjx-eqn-pointLightLuminousPower

    // NOTE: (*light).color.rgb is premultiplied with (*light).intensity / 4 π (which would be the luminous intensity) on the CPU

    var color: vec3<f32>;
#ifdef STANDARD_MATERIAL_CLEARCOAT
    // Account for the Fresnel term from the clearcoat darkening the main layer.
    //
    // <https://google.github.io/filament/Filament.html#materialsystem/clearcoatmodel/integrationinthesurfaceresponse>
    color = (diffuse + specular_light * inv_Fc) * inv_Fc + Frc;
#else   // STANDARD_MATERIAL_CLEARCOAT
    color = diffuse + specular_light;
#endif  // STANDARD_MATERIAL_CLEARCOAT

    return color * light_color * derived_input.NdotL;
}


fn apply_screen_lighting(in: pbr_types::PbrInput, out_color: ptr<function, vec4<f32>>) {
    var output_color: vec4<f32> = in.material.base_color;

    let emissive = in.material.emissive;

    // calculate non-linear roughness from linear perceptualRoughness
    let metallic = in.material.metallic;
    let perceptual_roughness = in.material.perceptual_roughness;
    let roughness = lighting::perceptualRoughnessToRoughness(perceptual_roughness);
    let ior = in.material.ior;
    let thickness = in.material.thickness;
    let reflectance = in.material.reflectance;
    let diffuse_transmission = in.material.diffuse_transmission;
    let specular_transmission = in.material.specular_transmission;

    let specular_transmissive_color = specular_transmission * in.material.base_color.rgb;

    let diffuse_occlusion = in.diffuse_occlusion;
    let specular_occlusion = in.specular_occlusion;

    // Neubelt and Pettineo 2013, "Crafting a Next-gen Material Pipeline for The Order: 1886"
    let NdotV = max(dot(in.N, in.V), 0.0001);
    let R = reflect(-in.V, in.N);

#ifdef STANDARD_MATERIAL_CLEARCOAT
    // Do the above calculations again for the clearcoat layer. Remember that
    // the clearcoat can have its own roughness and its own normal.
    let clearcoat = in.material.clearcoat;
    let clearcoat_perceptual_roughness = in.material.clearcoat_perceptual_roughness;
    let clearcoat_roughness = lighting::perceptualRoughnessToRoughness(clearcoat_perceptual_roughness);
    let clearcoat_N = in.clearcoat_N;
    let clearcoat_NdotV = max(dot(clearcoat_N, in.V), 0.0001);
    let clearcoat_R = reflect(-in.V, clearcoat_N);
#endif  // STANDARD_MATERIAL_CLEARCOAT

    let diffuse_color = pbr_functions::calculate_diffuse_color(
        output_color.rgb,
        metallic,
        specular_transmission,
        diffuse_transmission
    );

    // Diffuse transmissive strength is inversely related to metallicity and specular transmission, but directly related to diffuse transmission
    let diffuse_transmissive_color = output_color.rgb * (1.0 - metallic) * (1.0 - specular_transmission) * diffuse_transmission;

    // Calculate the world position of the second Lambertian lobe used for diffuse transmission, by subtracting material thickness
    let diffuse_transmissive_lobe_world_position = in.world_position - vec4<f32>(in.world_normal, 0.0) * thickness;

    let F0 = pbr_functions::calculate_F0(output_color.rgb, metallic, reflectance);
    let F_ab = lighting::F_AB(perceptual_roughness, NdotV);

    var lighting_input: lighting::LightingInput;
    lighting_input.layers[LAYER_BASE].NdotV = NdotV;
    lighting_input.layers[LAYER_BASE].N = in.N;
    lighting_input.layers[LAYER_BASE].R = R;
    lighting_input.layers[LAYER_BASE].perceptual_roughness = perceptual_roughness;
    lighting_input.layers[LAYER_BASE].roughness = roughness;
    lighting_input.P = in.world_position.xyz;
    lighting_input.V = in.V;
    lighting_input.diffuse_color = diffuse_color;
    lighting_input.F0_ = F0;
    lighting_input.F_ab = F_ab;
#ifdef STANDARD_MATERIAL_CLEARCOAT
    lighting_input.layers[LAYER_CLEARCOAT].NdotV = clearcoat_NdotV;
    lighting_input.layers[LAYER_CLEARCOAT].N = clearcoat_N;
    lighting_input.layers[LAYER_CLEARCOAT].R = clearcoat_R;
    lighting_input.layers[LAYER_CLEARCOAT].perceptual_roughness = clearcoat_perceptual_roughness;
    lighting_input.layers[LAYER_CLEARCOAT].roughness = clearcoat_roughness;
    lighting_input.clearcoat_strength = clearcoat;
#endif  // STANDARD_MATERIAL_CLEARCOAT
#ifdef STANDARD_MATERIAL_ANISOTROPY
    lighting_input.anisotropy = in.anisotropy_strength;
    lighting_input.Ta = in.anisotropy_T;
    lighting_input.Ba = in.anisotropy_B;
#endif  // STANDARD_MATERIAL_ANISOTROPY

    // And do the same for transmissive if we need to.
#ifdef STANDARD_MATERIAL_DIFFUSE_TRANSMISSION
    var transmissive_lighting_input: lighting::LightingInput;
    transmissive_lighting_input.layers[LAYER_BASE].NdotV = 1.0;
    transmissive_lighting_input.layers[LAYER_BASE].N = -in.N;
    transmissive_lighting_input.layers[LAYER_BASE].R = vec3(0.0);
    transmissive_lighting_input.layers[LAYER_BASE].perceptual_roughness = 1.0;
    transmissive_lighting_input.layers[LAYER_BASE].roughness = 1.0;
    transmissive_lighting_input.P = diffuse_transmissive_lobe_world_position.xyz;
    transmissive_lighting_input.V = -in.V;
    transmissive_lighting_input.diffuse_color = diffuse_transmissive_color;
    transmissive_lighting_input.F0_ = vec3(0.0);
    transmissive_lighting_input.F_ab = vec2(0.1);
#ifdef STANDARD_MATERIAL_CLEARCOAT
    transmissive_lighting_input.layers[LAYER_CLEARCOAT].NdotV = 0.0;
    transmissive_lighting_input.layers[LAYER_CLEARCOAT].N = vec3(0.0);
    transmissive_lighting_input.layers[LAYER_CLEARCOAT].R = vec3(0.0);
    transmissive_lighting_input.layers[LAYER_CLEARCOAT].perceptual_roughness = 0.0;
    transmissive_lighting_input.layers[LAYER_CLEARCOAT].roughness = 0.0;
    transmissive_lighting_input.clearcoat_strength = 0.0;
#endif  // STANDARD_MATERIAL_CLEARCOAT
#ifdef STANDARD_MATERIAL_ANISOTROPY
    lighting_input.anisotropy = in.anisotropy_strength;
    lighting_input.Ta = in.anisotropy_T;
    lighting_input.Ba = in.anisotropy_B;
#endif  // STANDARD_MATERIAL_ANISOTROPY
#endif  // STANDARD_MATERIAL_DIFFUSE_TRANSMISSION

    let light_color = fetch_screen_light_color(
        screen_light.light_pos, 
        screen_light.forward_dir,
        in.world_position.xyz,
        in.world_normal,
        shadow_map,
        view_bindings::directional_shadow_textures_comparison_sampler
    );
    let light_contrib = screen_light_contrib(screen_light.light_pos, light_color, &lighting_input);

    // i should be multiplying by exposure here i think? but that made there be basically no color so idk
    *out_color += vec4(light_contrib, 0.0);
}

struct ScreenLightUniform {
    clip_from_world: mat4x4<f32>,
    image_size: vec2<u32>,
    forward_dir: vec3<f32>,
    light_pos: vec3<f32>,
}

@group(2) @binding(96) var<uniform> screen_light: ScreenLightUniform;
@group(2) @binding(97) var shadow_map: texture_depth_2d;
@group(2) @binding(98) var screen_image: texture_2d<f32>;
@group(2) @binding(99) var screen_sampler: sampler;

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {

    // generate a PbrInput struct from the StandardMaterial bindings
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    // alpha discard
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

#ifdef PREPASS_PIPELINE
    let out = deferred_output(in, pbr_input);
#else
    var out: FragmentOutput;
    // apply lighting
    out.color = apply_pbr_lighting(pbr_input);

    var screen_contrib: vec4<f32> = vec4<f32>(0.0);
    apply_screen_lighting(pbr_input, &screen_contrib);
    out.color += screen_contrib;

    // apply in-shader post processing (fog, alpha-premultiply, and also tonemapping, debanding if the camera is non-hdr)
    // note this does not include fullscreen postprocessing effects like bloom.
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);

#endif
    return out;
}