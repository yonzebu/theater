//! This module is currently a little broken, it seems like video and audio desync and I'm not
//! currently sure why. However, I consider this project "done" anyways and don't plan on fixing
//! it unless I decide to reuse it.
use std::{
    collections::VecDeque,
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::{
        mpsc::{channel, Receiver, Sender, TryRecvError},
        Arc, Mutex, PoisonError,
    },
};

use bevy::{
    asset::{AssetLoader, RenderAssetUsages},
    audio::{self, AddAudioSource},
    core_pipeline::fullscreen_vertex_shader::fullscreen_shader_vertex_state,
    ecs::system::{lifetimeless::SResMut, StaticSystemParam, SystemParam},
    image::ImageSampler,
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_phase::TrackedRenderPass,
        render_resource::{
            binding_types::{sampler, texture_2d},
            BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, BlendState,
            ColorTargetState, ColorWrites, CommandEncoderDescriptor, Extent3d, FilterMode,
            FragmentState, LoadOp, MultisampleState, Operations, PipelineCache, PrimitiveState,
            RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor, Sampler,
            SamplerBindingType, SamplerDescriptor, ShaderStages, SpecializedRenderPipeline,
            SpecializedRenderPipelines, StoreOp, TextureAspect, TextureDescriptor,
            TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
            TextureViewDescriptor,
        },
        renderer::{render_system, RenderDevice, RenderQueue},
        texture::GpuImage,
        Render, RenderApp, RenderSet,
    },
    tasks::AsyncComputeTaskPool,
    utils::HashSet,
};
use ffmpeg_next::{
    self as ffmpeg, codec, decoder,
    ffi::EAGAIN,
    format::{context::Input, Pixel, Sample},
    frame, media,
    software::scaling,
    Packet, Rational,
};

const VIDEO_LOOKAHEAD: f64 = 2.0;
const AUDIO_LOOKAHEAD: f64 = 2.0;

struct ScalingContext(scaling::Context);
// SAFETY: i'm like 90% sure this is fine
unsafe impl Send for ScalingContext {}

#[derive(Debug)]
struct DecodedFrame {
    width: u32,
    height: u32,
    data: Vec<u8>,
    pts: i64,
    mip_levels: u32,
}

impl From<DecodedFrame> for Image {
    fn from(value: DecodedFrame) -> Self {
        let mut usage = TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
        if value.mip_levels > 1 {
            usage |= TextureUsages::RENDER_ATTACHMENT;
            //     let mut mip_level = 1;
            //     let mut width = value.width as usize;
            //     let mut height = value.height as usize;
            //     let mut current_offset = 0;
            //     while width > 1 || height > 1 || mip_level < value.mip_levels {
            //         let old_height = width;
            //         let old_width = width;
            //         width /= 2;
            //         width = width.max(1);
            //         height /= 2;
            //         height = height.max(1);
            //         let next_offset = data.len();
            //         data.reserve(width * height * 4);
            //         for y in 0..height {
            //             for x in 0..width {
            //                 let tl_start = current_offset + 8 * y * old_height + 8 * x;
            //                 let tl = data.get(tl_start..tl_start + 4).unwrap_or(&[0, 0, 0, 0]);
            //                 let tr_start = current_offset + 8 * y * old_height + 8 * x + 4;
            //                 let tr = data.get(tr_start..tr_start + 4).unwrap_or(&[0, 0, 0, 0]);
            //                 let bl_start = current_offset + (8 * y + 4) * old_height + 8 * x;
            //                 let bl = data.get(bl_start..bl_start + 4).unwrap_or(&[0, 0, 0, 0]);
            //                 let br_start = current_offset + (8 * y + 4) * old_height + 8 * x + 4;
            //                 let br = data.get(br_start..br_start + 4).unwrap_or(&[0, 0, 0, 0]);

            //                 data.extend([
            //                     (((tl[0] as u32 + bl[0] as u32) / 2 + (tr[0] as u32 + br[0] as u32) / 2) / 2) as u8,
            //                     (((tl[1] as u32 + bl[1] as u32) / 2 + (tr[1] as u32 + br[1] as u32) / 2) / 2) as u8,
            //                     (((tl[2] as u32 + bl[2] as u32) / 2 + (tr[2] as u32 + br[2] as u32) / 2) / 2) as u8,
            //                     (((tl[3] as u32 + bl[3] as u32) / 2 + (tr[3] as u32 + br[3] as u32) / 2) / 2) as u8,
            //                 ]);
            //             }
            //         }
            //         mip_level += 1;
            //         current_offset = next_offset;
            //     }
        }
        Image {
            data: value.data,
            texture_descriptor: TextureDescriptor {
                label: None,
                size: Extent3d {
                    width: value.width,
                    height: value.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: value.mip_levels.max(1),
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage,
                view_formats: &[],
            },
            sampler: ImageSampler::linear(),
            texture_view_descriptor: Some(TextureViewDescriptor {
                label: None,
                format: None,
                dimension: None,
                aspect: TextureAspect::All,
                base_mip_level: 0,
                mip_level_count: None,
                base_array_layer: 0,
                array_layer_count: None,
            }),
            asset_usage: RenderAssetUsages::all(),
        }
    }
}

// assumes all slices have the same length
struct InterleaveChannels<'a, F: 'a + FnMut(usize) -> &'a [T], T: 'a> {
    get_channel_data: F,
    channels: usize,
    current_channel: usize,
    current_elem: usize,
    elems: usize,
}

impl<'a, F: 'a + FnMut(usize) -> &'a [T], T: 'a> InterleaveChannels<'a, F, T> {
    fn new(mut get_channel_data: F, channels: usize) -> Self {
        let elems = get_channel_data(0).len();
        for c in 0..channels {
            debug_assert_eq!(elems, get_channel_data(c).len());
        }
        InterleaveChannels {
            get_channel_data,
            channels,
            current_channel: 0,
            current_elem: 0,
            elems,
        }
    }
}

impl<'a, F: 'a + FnMut(usize) -> &'a [T], T: 'a> Iterator for InterleaveChannels<'a, F, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_elem >= self.elems {
            return None;
        }
        let channel = self.current_channel;
        self.current_channel = self.current_channel.wrapping_add(1) % self.channels;
        let elem = self.current_elem;
        if self.current_channel == 0 {
            self.current_elem += 1;
        }
        Some(&(self.get_channel_data)(channel)[elem])
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.current_elem >= self.elems {
            return (0, Some(0));
        }
        let remaining = self.channels * (self.elems - self.current_elem) - self.current_channel;
        (remaining, Some(remaining))
    }
}

#[derive(Clone, Debug)]
struct DecodedAudio {
    /// the latest pts of the contained samples
    pts: i64,
    samples: Vec<f32>,
}

#[derive(Debug)]
pub struct MipInfo {
    pub mip_count: u32,
    pub data_len: usize,
}

fn calculate_mip_info(
    mut width: u32,
    mut height: u32,
    texel_size: u32,
    use_mips: bool,
    // ideally this should account for max mips too i think but i probably don't need it for this specific thing
) -> MipInfo {
    let mut len = width as usize * height as usize * texel_size as usize;
    let mut count = 1;
    if !use_mips {
        return MipInfo {
            data_len: len,
            mip_count: count,
        };
    }
    while width > 1 || height > 1 {
        width /= 2;
        width = width.max(1);
        height /= 2;
        height = height.max(1);
        len += width as usize * height as usize * texel_size as usize;
        count += 1;
    }
    MipInfo {
        data_len: len,
        mip_count: count,
    }
}

struct VideoDecoder {
    input: Input,
    video_stream_index: usize,
    audio_stream_index: usize,
    video_decoder: decoder::Video,
    audio_decoder: decoder::Audio,
    scaler: ScalingContext,
    latest_video_pts: Option<i64>,
    latest_audio_pts: Option<i64>,
    finished: bool,
    send_frames: Sender<DecodedFrame>,
    send_audio: Sender<DecodedAudio>,
    send_self: Sender<VideoDecoder>,
    use_mips: bool,
}
// SAFETY: i don't plan on accessing this when i don't have mutable access anyways
unsafe impl Sync for VideoDecoder {}

impl VideoDecoder {
    fn decode_until_pts_targets(mut self, video_target: i64, audio_target: i64) {
        while self.latest_video_pts.is_none_or(|pts| pts <= video_target)
            || self.latest_audio_pts.is_none_or(|pts| pts <= audio_target)
        {
            let mut should_send_packet;
            let mut packet;
            let is_audio_packet;
            loop {
                packet = Packet::empty();
                match packet.read(&mut self.input) {
                    Ok(()) => should_send_packet = true,
                    Err(ffmpeg::Error::Other { errno: EAGAIN }) => should_send_packet = false,
                    Err(ffmpeg::Error::Eof) => {
                        self.finished = true;
                        self.finish_task();
                        return;
                    }
                    Err(e) => panic!("packet read error: {e:?}"),
                }
                if packet.stream() == self.video_stream_index {
                    is_audio_packet = false;
                    break;
                }
                if packet.stream() == self.audio_stream_index {
                    is_audio_packet = true;
                    break;
                }
            }
            if should_send_packet {
                if is_audio_packet {
                    if let Err(e) = self.audio_decoder.send_packet(&packet) {
                        warn!("send audio packet error: {e:?}");
                    }
                } else if let Err(e) = self.video_decoder.send_packet(&packet) {
                    warn!("send video packet error: {e:?}");
                }
            }

            if is_audio_packet {
                // decode audio frame(s)
                let mut decoded;
                let mut samples = Vec::new();
                loop {
                    decoded = frame::Audio::empty();
                    match self.audio_decoder.receive_frame(&mut decoded) {
                        Ok(()) => {
                            self.latest_audio_pts = decoded.pts();
                            // i wish there was a nicer way to do this (there probably is)
                            match self.audio_decoder.format() {
                                Sample::None => panic!("unable to determine audio sample format"),
                                Sample::U8(_) => {
                                    samples.extend(
                                        InterleaveChannels::new(
                                            |c| decoded.plane::<u8>(c),
                                            decoded.planes(),
                                        )
                                        .map(|&s| 2. * (s as f32 / u8::MAX as f32) - 1.),
                                    );
                                }
                                Sample::I16(_) => {
                                    samples.extend(
                                        InterleaveChannels::new(
                                            |c| decoded.plane::<i16>(c),
                                            decoded.planes(),
                                        )
                                        .map(|&s| s as f32 / i16::MAX as f32),
                                    );
                                }
                                Sample::I32(_) => {
                                    samples.extend(
                                        InterleaveChannels::new(
                                            |c| decoded.plane::<i32>(c),
                                            decoded.planes(),
                                        )
                                        .map(|&s| s as f32 / i32::MAX as f32),
                                    );
                                }
                                Sample::I64(_) => {
                                    // this is kinda a hack bc apparently Sample isn't implemented for i64
                                    samples.extend(
                                        InterleaveChannels::new(
                                            |c| decoded.plane::<f64>(c),
                                            decoded.planes(),
                                        )
                                        .map(|&s| s.to_bits() as i64 as f32 / i64::MAX as f32),
                                    );
                                }
                                Sample::F32(_) => {
                                    samples.extend(InterleaveChannels::new(
                                        |c| decoded.plane::<f32>(c),
                                        decoded.planes(),
                                    ));
                                }
                                Sample::F64(_) => {
                                    samples.extend(
                                        InterleaveChannels::new(
                                            |c| decoded.plane::<f64>(c),
                                            decoded.planes(),
                                        )
                                        .map(|&s| s as f32),
                                    );
                                }
                            }
                        }
                        Err(ffmpeg::Error::Other { errno: EAGAIN }) => break,
                        Err(e) => panic!("receive frame error: {e:?}"),
                    }
                }
                let _ = self.send_audio.send(DecodedAudio {
                    pts: self.latest_audio_pts.unwrap(),
                    samples,
                });
            } else {
                // decode video frame(s)
                let mut decoded;
                let mut scaled;
                loop {
                    decoded = frame::Video::empty();
                    match self.video_decoder.receive_frame(&mut decoded) {
                        Ok(()) => {
                            self.latest_video_pts = decoded.pts();
                            scaled = frame::Video::empty();
                            self.scaler.0.run(&decoded, &mut scaled).unwrap();

                            let mut data = scaled.data(0).to_owned();
                            let mip_info = calculate_mip_info(
                                scaled.width(),
                                scaled.height(),
                                4,
                                self.use_mips,
                            );
                            if !self.use_mips {
                                debug_assert_eq!(data.len(), mip_info.data_len);
                            }
                            data.resize(mip_info.data_len, 0);

                            let _ = self.send_frames.send(DecodedFrame {
                                width: scaled.width(),
                                height: scaled.height(),
                                data,
                                pts: decoded.pts().unwrap(),
                                mip_levels: mip_info.mip_count,
                            });
                        }
                        Err(ffmpeg::Error::Other { errno: EAGAIN }) => break,
                        Err(e) => panic!("receive frame error: {e:?}"),
                    }
                }
            }
        }

        self.finish_task();
    }

    fn finish_task(self) {
        let send_self = self.send_self.clone();
        let _ = send_self.send(self);
    }
}

#[derive(Clone)]
struct BufferedAudio {
    pts: i64,
    samples: Arc<[f32]>,
}

#[derive(Debug)]
pub struct BufferedFrame {
    pub image: Handle<Image>,
    pts: i64,
}

#[derive(Asset, TypePath)]
pub struct VideoStream {
    decoder: Option<VideoDecoder>,
    recv_decoder: Mutex<Receiver<VideoDecoder>>,
    recv_frames: Mutex<Receiver<DecodedFrame>>,
    recv_audio: Mutex<Receiver<DecodedAudio>>,
    /// audio is accessed with the [`VideoAudioSource`]
    buffered_audio: VecDeque<BufferedAudio>,
    audio_sinks: Mutex<Vec<Sender<BufferedAudio>>>,
    /// frames that are currently loaded and available to the renderer
    pub buffered_frames: VecDeque<BufferedFrame>,
    pub playing: bool,
    /// in seconds (i think)
    video_progress: Rational,
    video_time_base: Rational,
    /// in seconds (i think)
    audio_progress: Rational,
    audio_time_base: Rational,
    audio_channels: u16,
    audio_sample_rate: u32,
    finished_decoding: bool,
    finished: bool,
}

impl VideoStream {
    pub fn new(source: impl AsRef<Path>, use_mips: bool) -> Result<Self, ffmpeg::Error> {
        let input = ffmpeg::format::input(&source)?;
        let video_stream = input
            .streams()
            .best(media::Type::Video)
            .ok_or(ffmpeg::Error::StreamNotFound)?;
        let audio_stream = input
            .streams()
            .best(media::Type::Audio)
            .ok_or(ffmpeg::Error::StreamNotFound)?;
        info!(
            "video stream stuff: time_base = {:?}, start_time={}, duration = {}, frames = {}, rate = {:?}, avg frame rate = {:?}",
            video_stream.time_base(), video_stream.start_time(), video_stream.duration(), video_stream.frames(), video_stream.rate(), video_stream.avg_frame_rate()
        );
        info!(
            "audio stream stuff: time_base = {:?}, start_time={}, duration = {}, frames = {}, sample rate = {:?}, avg frame rate = {:?}",
            audio_stream.time_base(), audio_stream.start_time(), audio_stream.duration(), audio_stream.frames(), audio_stream.rate(), audio_stream.avg_frame_rate(),
        );
        let video_time_base = video_stream.time_base();
        let video_start_time = video_stream.start_time();
        let video_stream_index = video_stream.index();
        let video_codec_ctx = codec::Context::from_parameters(video_stream.parameters())?;
        let video_decoder = video_codec_ctx.decoder().video()?;
        let scaler = ScalingContext(video_decoder.converter(Pixel::RGBA)?);
        let audio_stream_index = audio_stream.index();
        let audio_time_base = audio_stream.time_base();
        let audio_start_time = audio_stream.start_time();
        let audio_codec_ctx = codec::Context::from_parameters(audio_stream.parameters())?;
        let audio_decoder = audio_codec_ctx.decoder().audio()?;
        let audio_channels = audio_decoder.channels();
        let audio_sample_rate = audio_decoder.rate();
        let (send_frames, recv_frames) = channel();
        let (send_decoder, recv_decoder) = channel();
        let (send_audio, recv_audio) = channel();
        Ok(VideoStream {
            decoder: Some(VideoDecoder {
                input,
                video_stream_index,
                audio_stream_index,
                video_decoder,
                audio_decoder,
                scaler,
                latest_audio_pts: None,
                latest_video_pts: None,
                finished: false,
                send_self: send_decoder,
                send_frames,
                send_audio,
                use_mips,
            }),
            recv_decoder: Mutex::new(recv_decoder),
            recv_frames: Mutex::new(recv_frames),
            recv_audio: Mutex::new(recv_audio),
            buffered_audio: VecDeque::new(),
            audio_sinks: Mutex::new(vec![]),
            buffered_frames: VecDeque::new(),
            playing: false,
            video_progress: Rational::new(video_start_time as i32, 1) * video_time_base,
            video_time_base,
            audio_progress: Rational::new(audio_start_time as i32, 1) * audio_time_base,
            audio_time_base,
            audio_channels,
            audio_sample_rate,
            finished_decoding: false,
            finished: false,
        })
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }
}

pub struct VideoAudioSource {
    current: Option<BufferedAudio>,
    buffered_audio: VecDeque<BufferedAudio>,
    recv_audio: Receiver<BufferedAudio>,
    current_index: usize,
    channels: u16,
    sample_rate: u32,
}

impl Iterator for VideoAudioSource {
    type Item = f32;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(&sample) = self
                .current
                .as_ref()
                .and_then(|current| current.samples.get(self.current_index))
            {
                self.current_index += 1;
                return Some(sample);
            }
            self.current_index = 0;
            while let Some(new_current) = self.buffered_audio.pop_front() {
                if new_current.samples.len() > 0 {
                    let sample = new_current.samples[0];
                    self.current = Some(new_current);
                    self.current_index += 1;
                    return Some(sample);
                }
            }
            loop {
                match self.recv_audio.try_recv() {
                    Ok(audio) => {
                        self.buffered_audio.push_back(audio);
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => return None,
                }
            }
        }
    }
}

impl audio::Source for VideoAudioSource {
    fn channels(&self) -> u16 {
        self.channels
    }
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    fn current_frame_len(&self) -> Option<usize> {
        None
    }
    fn total_duration(&self) -> Option<std::time::Duration> {
        None
    }
}

impl Decodable for VideoStream {
    type DecoderItem = f32;
    type Decoder = VideoAudioSource;
    fn decoder(&self) -> Self::Decoder {
        let (send, recv) = channel();
        self.audio_sinks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(send);
        VideoAudioSource {
            buffered_audio: self.buffered_audio.clone(),
            recv_audio: recv,
            current: None,
            current_index: 0,
            channels: self.audio_channels,
            sample_rate: self.audio_sample_rate,
        }
    }
}

#[derive(Clone, Default, Deref, DerefMut, Resource, ExtractResource, Reflect)]
struct QueuedFrameMips(Vec<AssetId<Image>>);

#[derive(Resource)]
pub struct MipGenerationPipeline {
    pub texture_bind_group: BindGroupLayout,
    pub sampler: Sampler,
    pub shader: Handle<Shader>,
}

impl FromWorld for MipGenerationPipeline {
    fn from_world(render_world: &mut World) -> Self {
        let asset_server = render_world.resource::<AssetServer>();

        let shader = asset_server.load("generate_mips.wgsl");

        let render_device = render_world.resource::<RenderDevice>();

        let texture_bind_group = render_device.create_bind_group_layout(
            "mip_generation_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    sampler(SamplerBindingType::Filtering),
                ),
            ),
        );

        let sampler = render_device.create_sampler(&SamplerDescriptor {
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..default()
        });

        MipGenerationPipeline {
            texture_bind_group,
            sampler,
            shader,
        }
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub struct MipGenerationPipelineKey {
    pub texture_format: TextureFormat,
    pub blend_state: Option<BlendState>,
    pub samples: u32,
}

impl SpecializedRenderPipeline for MipGenerationPipeline {
    type Key = MipGenerationPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("mip generation pipeline".into()),
            layout: vec![self.texture_bind_group.clone()],
            vertex: fullscreen_shader_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs: vec![],
                entry_point: "fragment".into(),
                targets: vec![Some(ColorTargetState {
                    format: key.texture_format,
                    blend: key.blend_state,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState {
                count: key.samples,
                ..Default::default()
            },
            push_constant_ranges: Vec::new(),
            zero_initialize_workgroup_memory: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Event, Deref, DerefMut)]
#[non_exhaustive]
pub struct VideoFinished(pub AssetId<VideoStream>);

#[derive(Default, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct VideoStreamSettings {
    pub use_mips: bool,
}

struct VideoStreamLoader {
    root: PathBuf,
}

impl FromWorld for VideoStreamLoader {
    fn from_world(_world: &mut World) -> Self {
        // this is a hack that depends on using a filesystem-based asset loader
        // it's copied from how bevy's filesystem-based asset loader determines the assets directory
        // it sucks but bevy doesn't give the information the ffmpeg wrapper wants (file path) with
        // the general asset loading
        // you could drop into the raw ffi and do custom io i think, though, i just didn't need to
        // for this project
        // also i'm not sure that would work for my use case (streaming a large asset), since bevy's
        // asset loading seems like it expects assets to be loaded "all at once".
        let root = if let Ok(manifest_dir) = std::env::var("BEVY_ASSET_ROOT") {
            PathBuf::from(manifest_dir)
        } else if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
            PathBuf::from(manifest_dir)
        } else {
            std::env::current_exe()
                .map(|path| path.parent().map(ToOwned::to_owned).unwrap())
                .unwrap()
        }
        .join("assets");
        VideoStreamLoader { root }
    }
}

impl AssetLoader for VideoStreamLoader {
    type Asset = VideoStream;
    type Error = ffmpeg::Error;
    type Settings = VideoStreamSettings;
    async fn load(
        &self,
        _reader: &mut dyn bevy::asset::io::Reader,
        settings: &Self::Settings,
        load_context: &mut bevy::asset::LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        VideoStream::new(self.root.join(load_context.path()), settings.use_mips)
    }
    fn extensions(&self) -> &[&str] {
        &["mp4"]
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, SystemSet)]
pub enum VideoSet {
    UpdateStreams,
    UpdatePlayers,
}

pub struct VideoPlugin;

impl Plugin for VideoPlugin {
    fn build(&self, app: &mut App) {
        ffmpeg::init().unwrap();
        app.init_asset::<VideoStream>()
            .init_asset_loader::<VideoStreamLoader>()
            .add_audio_source::<VideoStream>()
            .add_event::<VideoFinished>()
            .init_resource::<QueuedFrameMips>()
            .add_systems(PostUpdate, update_videos.in_set(VideoSet::UpdateStreams))
            .configure_sets(
                PostUpdate,
                VideoSet::UpdateStreams.before(VideoSet::UpdatePlayers),
            )
            .add_plugins(ExtractResourcePlugin::<QueuedFrameMips>::default());
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<RetryNextFrameMips>()
                .add_systems(
                    Render,
                    (
                        queue_mip_pipelines.in_set(RenderSet::PrepareResources),
                        // this is after the render system so the pipeline cache knows about whatever got queued, but it does mean that there will be definitely at least 1 frame per image without the updated mips
                        queue_frame_mips_generation
                            .after(render_system)
                            .in_set(RenderSet::Render),
                    ),
                );
        }
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<MipGenerationPipeline>()
                .init_resource::<SpecializedRenderPipelines<MipGenerationPipeline>>();
        }
    }
}

#[derive(Component)]
pub struct VideoPlayer(pub Handle<VideoStream>);

pub trait ReceiveFrame {
    type Param: SystemParam + 'static;
    /// Used to avoid triggering potentially costly mutation when mutation is unnecessary (for
    /// instance, constantly copying image handles to a [`StandardMaterial`] will constantly
    /// recreate the material bind groups, even if the image is the same image).
    #[inline]
    fn should_receive(
        &self,
        frame: &Handle<Image>,
        param: &<Self::Param as SystemParam>::Item<'_, '_>,
    ) -> bool {
        let _ = (frame, param);
        true
    }
    fn receive_frame(
        &mut self,
        frame: Handle<Image>,
        param: &mut <Self::Param as SystemParam>::Item<'_, '_>,
    );
}
impl ReceiveFrame for StandardMaterial {
    type Param = ();
    #[inline]
    fn should_receive(
        &self,
        frame: &Handle<Image>,
        _: &<Self::Param as SystemParam>::Item<'_, '_>,
    ) -> bool {
        self.base_color_texture
            .as_ref()
            .map_or(true, |h| h.id() != frame.id())
    }
    #[inline]
    fn receive_frame(
        &mut self,
        frame: Handle<Image>,
        _: &mut <Self::Param as SystemParam>::Item<'_, '_>,
    ) {
        self.base_color_texture = Some(frame)
    }
}
impl<M: Material + ReceiveFrame> ReceiveFrame for MeshMaterial3d<M> {
    type Param = (SResMut<Assets<M>>, <M as ReceiveFrame>::Param);
    fn should_receive(
        &self,
        frame: &Handle<Image>,
        (materials, material_param): &<Self::Param as SystemParam>::Item<'_, '_>,
    ) -> bool {
        materials.get(self.0.id()).map_or(false, |material| {
            material.should_receive(frame, material_param)
        })
    }
    #[inline]
    fn receive_frame(
        &mut self,
        frame: Handle<Image>,
        (materials, material_param): &mut <Self::Param as SystemParam>::Item<'_, '_>,
    ) {
        if let Some(material) = materials.get_mut(self.0.id()) {
            material.receive_frame(frame, material_param);
        }
    }
}

pub struct VideoPlayerPlugin<R: ReceiveFrame>(PhantomData<R>);

impl<R: ReceiveFrame> Default for VideoPlayerPlugin<R> {
    fn default() -> Self {
        VideoPlayerPlugin(PhantomData)
    }
}

impl<R: ReceiveFrame + Component> Plugin for VideoPlayerPlugin<R> {
    fn build(&self, app: &mut App) {
        app.add_systems(
            PostUpdate,
            update_receivers_from_players::<R>.in_set(VideoSet::UpdatePlayers),
        );
    }
}

fn update_videos(
    mut commands: Commands,
    mut videos: ResMut<Assets<VideoStream>>,
    mut images: ResMut<Assets<Image>>,
    mut queued_frame_mips: ResMut<QueuedFrameMips>,
    mut video_finisheds: EventWriter<VideoFinished>,
    time: Res<Time>,
) {
    queued_frame_mips.clear();
    let dt: Rational = time.delta_secs_f64().into();
    for (video_id, video) in videos.iter_mut() {
        let video = &mut *video;
        let vtb = video.video_time_base;
        let atb = video.audio_time_base;
        if video.playing {
            video.video_progress = video.video_progress + dt;
            video.audio_progress = video.audio_progress + dt;
        }
        let video_lookahead = video.video_progress + Into::<Rational>::into(VIDEO_LOOKAHEAD);
        let audio_lookahead = video.audio_progress + Into::<Rational>::into(AUDIO_LOOKAHEAD);
        if let Some(mut decoder) = video.decoder.take() {
            let needs_frames = video
                .buffered_frames
                .back()
                .is_none_or(|frame| Rational(frame.pts as i32, 1) <= video_lookahead / vtb);
            let needs_audio = video
                .buffered_audio
                .back()
                .is_none_or(|audio| Rational(audio.pts as i32, 1) <= audio_lookahead / atb);
            if !decoder.finished && (needs_frames || needs_audio) {
                if let Some(frame) = video.buffered_frames.back() {
                    decoder.latest_video_pts = Some(frame.pts);
                }
                if let Some(audio) = video.buffered_audio.back() {
                    decoder.latest_audio_pts = Some(audio.pts);
                }
                AsyncComputeTaskPool::get()
                    .spawn(async move {
                        decoder.decode_until_pts_targets(
                            Into::<f64>::into(video_lookahead / vtb) as i64,
                            Into::<f64>::into(audio_lookahead / atb) as i64,
                        )
                    })
                    .detach();
            } else {
                if decoder.finished {
                    video
                        .audio_sinks
                        .get_mut()
                        .unwrap_or_else(PoisonError::into_inner)
                        .clear();
                    video.finished_decoding = true;
                }
                // no need to decode more frames yet
                video.decoder = Some(decoder);
            }
        } else if let Ok(decoder) = video
            .recv_decoder
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .try_recv()
        {
            video.decoder = Some(decoder);
        }

        while let Ok(frame) = video
            .recv_frames
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .try_recv()
        {
            let pts = frame.pts;
            let mip_levels = frame.mip_levels;
            let image: Image = frame.into();
            let image = images.add(image);
            let image_id = image.id();
            video
                .buffered_frames
                .push_back(BufferedFrame { image, pts });
            if mip_levels > 1 {
                queued_frame_mips.push(image_id);
            }
        }
        let mut first_pts = None;
        let mut all_samples = Vec::new();
        while let Ok(DecodedAudio { pts, mut samples }) = video
            .recv_audio
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .try_recv()
        {
            if all_samples.is_empty() {
                first_pts = Some(pts);
                std::mem::swap(&mut samples, &mut all_samples);
            } else {
                all_samples.extend(samples);
            }
        }
        if let Some(pts) = first_pts {
            let buffered = BufferedAudio {
                pts,
                samples: Arc::from(all_samples),
            };
            video
                .audio_sinks
                .get_mut()
                .unwrap()
                .retain(|sink| sink.send(buffered.clone()).is_ok());
            video.buffered_audio.push_back(buffered);
        }

        let current_vpts: f64 = (video.video_progress / vtb).into();
        let current_vpts = current_vpts as i64;
        let current = video
            .buffered_frames
            .partition_point(|frame| frame.pts < current_vpts);
        // at least one frame is always kept around so that the video doesn't just die if it gets behind
        if let Some(last) = video.buffered_frames.drain(0..current).last() {
            if video.buffered_frames.is_empty() {
                video.buffered_frames.push_back(last);
                if video.finished_decoding && !video.finished {
                    video.finished = true;
                    commands.trigger(VideoFinished(video_id));
                    video_finisheds.send(VideoFinished(video_id));
                }
            }
        }
        let current_apts: f64 = (video.audio_progress / atb).into();
        let current_apts = current_apts as i64;
        let current = video
            .buffered_audio
            .partition_point(|audio| audio.pts < current_apts);
        // idk this might be kinda scuffed to do for audio, playback loops are maybe not the best? but idk, i'm just gonna try to not worry about this right now
        if let Some(last) = video.buffered_audio.drain(0..current).last() {
            if video.buffered_audio.is_empty() {
                video.buffered_audio.push_back(last);
            }
        }
    }
}

fn queue_mip_pipelines(
    queued_frame_mips: Res<QueuedFrameMips>,
    pipeline_cache: Res<PipelineCache>,
    mut mip_pipelines: ResMut<SpecializedRenderPipelines<MipGenerationPipeline>>,
    mip_pipeline: Res<MipGenerationPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
) {
    for frame_id in queued_frame_mips.iter() {
        if let Some(gpu_image) = gpu_images.get(*frame_id) {
            mip_pipelines.specialize(
                &pipeline_cache,
                &mip_pipeline,
                MipGenerationPipelineKey {
                    texture_format: gpu_image.texture_format,
                    blend_state: None,
                    // this is probably incorrect in general but works in this particular case
                    samples: 1,
                },
            );
        }
    }
}

#[derive(Default, Deref, DerefMut, Resource, Reflect)]
struct RetryNextFrameMips(HashSet<AssetId<Image>>);

#[allow(clippy::too_many_arguments)]
fn queue_frame_mips_generation(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut queued_frame_mips: ResMut<QueuedFrameMips>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    mut mip_pipelines: ResMut<SpecializedRenderPipelines<MipGenerationPipeline>>,
    mip_pipeline: Res<MipGenerationPipeline>,
    pipeline_cache: Res<PipelineCache>,
    mut retry_next_frame_mips: ResMut<RetryNextFrameMips>,
) {
    let mut command_encoder = None;
    let mut try_queue_mips = |frame| {
        if let Some(gpu_image) = gpu_images.get(frame) {
            let encoder = command_encoder.get_or_insert_with(|| {
                render_device.create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("video frame mips generation"),
                })
            });
            let pipeline_id = mip_pipelines.specialize(
                &pipeline_cache,
                &mip_pipeline,
                MipGenerationPipelineKey {
                    texture_format: gpu_image.texture_format,
                    blend_state: None,
                    // this is probably incorrect in general but works in this particular case
                    samples: 1,
                },
            );
            let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline_id) else {
                return false;
            };
            let mip_levels =
                calculate_mip_info(gpu_image.size.x, gpu_image.size.y, 4, true).mip_count;
            let mut lower_mip = gpu_image.texture.create_view(&TextureViewDescriptor {
                label: None,
                format: None,
                dimension: None,
                aspect: TextureAspect::All,
                base_mip_level: 0,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: None,
            });
            let mut higher_mip;
            for mip in 1..mip_levels {
                higher_mip = gpu_image.texture.create_view(&TextureViewDescriptor {
                    label: None,
                    format: None,
                    dimension: None,
                    aspect: TextureAspect::All,
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: None,
                });
                let bind_group = render_device.create_bind_group(
                    "mip generation bind group",
                    &mip_pipeline.texture_bind_group,
                    &BindGroupEntries::sequential((&lower_mip, &mip_pipeline.sampler)),
                );
                let mut render_pass = TrackedRenderPass::new(
                    &render_device,
                    encoder.begin_render_pass(&RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(RenderPassColorAttachment {
                            view: &higher_mip,
                            resolve_target: None,
                            ops: Operations {
                                load: LoadOp::Clear(default()),
                                store: StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    }),
                );
                render_pass.set_render_pipeline(pipeline);
                render_pass.set_bind_group(0, &bind_group, &[]);
                render_pass.draw(0..3, 0..1);
                drop(render_pass);
                lower_mip = higher_mip;
            }
            return true;
        }
        false
    };
    retry_next_frame_mips.retain(|&queued_frame| !try_queue_mips(queued_frame));
    for queued_frame in queued_frame_mips.drain(..) {
        if !try_queue_mips(queued_frame) {
            retry_next_frame_mips.insert(queued_frame);
        }
    }
    if let Some(encoder) = command_encoder {
        let queue = render_queue.clone();
        AsyncComputeTaskPool::get()
            .spawn(async move {
                queue.submit([encoder.finish()]);
            })
            .detach();
    }
}

fn update_receivers_from_players<R: ReceiveFrame + Component>(
    mut video_players: Query<(&mut R, &VideoPlayer)>,
    mut video_streams: ResMut<Assets<VideoStream>>,
    param: StaticSystemParam<R::Param>,
) {
    let mut param = param.into_inner();
    for (mut receiver, player) in video_players.iter_mut() {
        if let Some(video_stream) = video_streams.get_mut(player.0.id()) {
            if let Some(frame) = video_stream
                .buffered_frames
                .front()
                .map(|frame| frame.image.clone())
            {
                if receiver.should_receive(&frame, &param) {
                    receiver.receive_frame(frame, &mut param);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::video::InterleaveChannels;

    #[test]
    fn test_interleave_channels() {
        let empties = &[&[], &[], &[], &[]];
        assert_eq!(
            InterleaveChannels::new(|c| empties[c], empties.len())
                .copied()
                .collect::<Vec<i32>>(),
            Vec::<i32>::new()
        );

        let channels = &[&[1, 2, 3], &[4, 5, 6], &[7, 8, 9]];
        assert_eq!(
            InterleaveChannels::new(|c| channels[c], channels.len())
                .copied()
                .collect::<Vec<_>>(),
            vec![1, 4, 7, 2, 5, 8, 3, 6, 9]
        );
    }
}
