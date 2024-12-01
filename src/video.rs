use std::{
    collections::VecDeque,
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::{
        mpsc::{channel, Receiver, RecvError, Sender, TryRecvError},
        Arc, Mutex, RwLock,
    },
};

use bevy::{
    asset::{AssetLoader, RenderAssetUsages},
    audio::{self, AddAudioSource},
    ecs::system::{lifetimeless::SResMut, StaticSystemParam, SystemParam},
    log::tracing_subscriber::filter::targets::Iter,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
    tasks::AsyncComputeTaskPool,
};
use ffmpeg_next::{
    self as ffmpeg, codec, decoder,
    ffi::EAGAIN,
    format::{context::Input, sample, Pixel, Sample},
    frame, media,
    software::scaling,
    Packet, Rational,
};
use zerocopy::FromBytes;

struct ScalingContext(scaling::Context);
// SAFETY: i'm like 90% sure this is fine
unsafe impl Send for ScalingContext {}

#[derive(Debug)]
struct DecodedFrame {
    width: u32,
    height: u32,
    data: Vec<u8>,
    pts: i64,
}

impl From<DecodedFrame> for Image {
    fn from(value: DecodedFrame) -> Self {
        Image::new(
            Extent3d {
                width: value.width,
                height: value.height,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            value.data,
            TextureFormat::Rgba8Unorm,
            RenderAssetUsages::all(),
        )
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

struct DecodedAudio {
    /// the latest pts of the contained samples
    pts: i64,
    samples: Arc<[f32]>,
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
                    let _ = self.audio_decoder.send_packet(&packet);
                } else {
                    let _ = self.video_decoder.send_packet(&packet);
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
                                        .map(|&s| {
                                            let s: i64 = zerocopy::transmute!(s);
                                            s as f32 / i64::MAX as f32
                                        }),
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
                    if decoded.rate() != self.audio_decoder.rate() {
                        println!(
                            "decoder rate: {}, decoded rate: {}",
                            self.audio_decoder.rate(),
                            decoded.rate()
                        );
                    }
                    if decoded.channels() != self.audio_decoder.channels() {
                        println!(
                            "decoder channels: {}, decoded channels: {}",
                            self.audio_decoder.channels(),
                            decoded.channels()
                        );
                    }
                }
                let _ = self.send_audio.send(DecodedAudio {
                    pts: self.latest_audio_pts.unwrap(),
                    samples: Arc::from(samples),
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

                            let _ = self.send_frames.send(DecodedFrame {
                                width: scaled.width(),
                                height: scaled.height(),
                                data: scaled.data(0).to_owned(),
                                pts: decoded.pts().unwrap(),
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
}

impl VideoStream {
    pub fn new(source: impl AsRef<Path>) -> Result<Self, ffmpeg::Error> {
        let input = ffmpeg::format::input(&source)?;
        let video_stream = input
            .streams()
            .best(media::Type::Video)
            .ok_or(ffmpeg::Error::StreamNotFound)?;
        let audio_stream = input
            .streams()
            .best(media::Type::Audio)
            .ok_or(ffmpeg::Error::StreamNotFound)?;
        println!(
            "video stream stuff: time_base = {:?}, start_time={}, duration = {}, frames = {}, rate = {:?}, avg frame rate = {:?}",
            video_stream.time_base(), video_stream.start_time(), video_stream.duration(), video_stream.frames(), video_stream.rate(), video_stream.avg_frame_rate()
        );
        println!(
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
        })
    }

    pub fn is_finished(&self) -> Option<bool> {
        self.decoder.as_ref().map(|decoder| decoder.finished)
    }

    // get new frames
    fn update(
        &mut self,
        mut convert_frame: impl FnMut(DecodedFrame) -> BufferedFrame,
        dt: Rational,
    ) {
        let vtb = self.video_time_base;
        let atb = self.audio_time_base;
        if self.playing {
            self.video_progress = self.video_progress + dt;
            self.audio_progress = self.audio_progress + dt;
        }
        let video_lookahead = self.video_progress + Rational(5, 1);
        let audio_lookahead = self.audio_progress + Rational(5, 1);
        if let Some(mut decoder) = self.decoder.take() {
            let needs_frames = self
                .buffered_frames
                .back()
                .is_none_or(|frame| Rational(frame.pts as i32, 1) <= video_lookahead / vtb);
            let needs_audio = self
                .buffered_audio
                .back()
                .is_none_or(|audio| Rational(audio.pts as i32, 1) <= audio_lookahead / atb);
            if (needs_frames || needs_audio) && !decoder.finished {
                if let Some(frame) = self.buffered_frames.back() {
                    decoder.latest_video_pts = Some(frame.pts);
                }
                if let Some(audio) = self.buffered_audio.back() {
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
                    self.audio_sinks.get_mut().unwrap().clear();
                }
                // no need to decode more frames yet
                self.decoder = Some(decoder);
            }
        } else {
            if let Ok(decoder) = self.recv_decoder.get_mut().unwrap().try_recv() {
                self.decoder = Some(decoder);
            }
        }

        while let Ok(frame) = self.recv_frames.get_mut().unwrap().try_recv() {
            self.buffered_frames.push_back(convert_frame(frame));
        }
        while let Ok(audio) = self.recv_audio.get_mut().unwrap().try_recv() {
            let buffered = BufferedAudio {
                pts: audio.pts,
                samples: audio.samples,
            };
            self.audio_sinks
                .get_mut()
                .unwrap()
                .retain(|sink| sink.send(buffered.clone()).is_ok());
            self.buffered_audio.push_back(buffered);
        }

        let current_vpts: f64 = (self.video_progress / vtb).into();
        let current_vpts = current_vpts as i64;
        let current = self
            .buffered_frames
            .partition_point(|frame| frame.pts < current_vpts);
        self.buffered_frames.drain(0..current);
        // i don't like how this is managed right now :/
        // i could probably find a way to both clean up old (unused) audio data + not make too many
        // unnecessary allocations, but it would be very annoying and probably not pretty
        // maybe if i drew in another dep though
        let current_apts: f64 = (self.audio_progress / atb).into();
        let current_apts = current_apts as i64;
        let current = self
            .buffered_audio
            .partition_point(|audio| audio.pts < current_apts);
        self.buffered_audio.drain(0..current);
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
            if let Some(current) = self.current.as_ref() {
                if let Some(&sample) = current.samples.get(self.current_index) {
                    self.current_index += 1;
                    return Some(sample);
                }
            }
            self.current_index = 0;
            while let Some(new_current) = self.buffered_audio.pop_front() {
                if new_current.samples.len() > 0 {
                    let sample = new_current.samples[0];
                    self.current = Some(new_current);
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
        // println!("channels: {}", self.channels);
        self.channels
    }
    fn sample_rate(&self) -> u32 {
        // println!("sample rate: {}", self.sample_rate);
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
        self.audio_sinks.lock().unwrap().push(send);
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
    type Settings = ();
    async fn load(
        &self,
        _reader: &mut dyn bevy::asset::io::Reader,
        _settings: &Self::Settings,
        load_context: &mut bevy::asset::LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        VideoStream::new(self.root.join(load_context.path()))
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
            .add_systems(PostUpdate, update_videos.in_set(VideoSet::UpdateStreams))
            .configure_sets(
                PostUpdate,
                VideoSet::UpdateStreams.before(VideoSet::UpdatePlayers),
            );
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
    mut videos: ResMut<Assets<VideoStream>>,
    mut images: ResMut<Assets<Image>>,
    time: Res<Time>,
) {
    for (_, video) in videos.iter_mut() {
        video.update(
            |frame| {
                let pts = frame.pts;
                BufferedFrame {
                    image: images.add(frame),
                    pts,
                }
            },
            time.delta_secs_f64().into(),
        );
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
                .get(0)
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
mod test {
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
