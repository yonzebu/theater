use std::{borrow::Cow, error::Error, fmt::Display, ops::Deref, time::Duration};

use bevy::{
    asset::{io::Reader, AssetLoader},
    ecs::{component::ComponentId, world::DeferredWorld},
    prelude::*,
};
use nom::{
    branch::alt,
    bytes::complete::{tag, take_till, take_till1},
    character::complete::{line_ending, multispace0},
    combinator::{eof, peek, verify},
    error::ParseError,
    multi::{fold_many0, many0},
    number::complete::double,
    sequence::{delimited, preceded, terminated, tuple},
    IResult, InputLength, Parser,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Answer {
    pub answer: String,
    pub response: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnswerBlock {
    /// a single answer, so only the one string is provided
    Single(String),
    /// multiple answers, each with a response
    Many(Vec<Answer>),
}

impl AnswerBlock {
    pub fn get_answer(&self, answer: usize) -> Option<&String> {
        match self {
            Self::Single(s) => (answer == 0).then_some(s),
            Self::Many(choices) => choices.get(answer).map(|answer| &answer.answer),
        }
    }
    pub fn get_response(&self, answer: usize) -> Option<&String> {
        match self {
            Self::Single(_) => None,
            Self::Many(choices) => choices.get(answer).map(|answer| &answer.response),
        }
    }
    pub fn answers_iter(&self) -> impl Iterator<Item = &String> {
        (0..)
            .map(|i| self.get_answer(i))
            .take_while(|answer| answer.is_some())
            .flatten()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScriptEntry {
    Prompt {
        prompt: String,
        choices: AnswerBlock,
    },
    Line(String),
    Wait(Duration),
    StartShow,
    /// Realistically no script runner SHOULD ever do anything with this, it's mostly a marker for parsing
    // maybe it's a good idea to have separate parsed entries vs. actual entries? if i had more time i might
    // rearchitect it that way
    OnShowEnd,
}

fn line_starter<'a>(starter: &'a str) -> impl FnMut(&'a str) -> IResult<&'a str, &'a str> {
    preceded(multispace0, tag(starter))
}

fn till_newline0<'a, E>(input: &'a str) -> IResult<&'a str, &'a str, E>
where
    E: ParseError<&'a str>,
{
    take_till(|c| c == '\r' || c == '\n')(input)
}

fn till_newline1<'a, E>(input: &'a str) -> IResult<&'a str, &'a str, E>
where
    E: ParseError<&'a str>,
{
    take_till1(|c| c == '\r' || c == '\n')(input)
}

fn command_line<'a, F, O, E>(f: F) -> impl FnMut(&'a str) -> IResult<&'a str, O, E>
where
    F: Parser<&'a str, O, E>,
    E: ParseError<&'a str>,
{
    delimited(
        tuple((multispace0, tag("["), multispace0)),
        f,
        tuple((
            multispace0,
            tag("]"),
            verify(till_newline0, |s: &str| s.trim().is_empty()),
            Parser::or(line_ending, eof),
        )),
    )
}

fn prompt(lines: &str) -> IResult<&str, &str> {
    delimited(
        line_starter(">"),
        till_newline1.map(str::trim),
        Parser::or(line_ending, eof),
    )(lines)
}

fn answer(lines: &str) -> IResult<&str, &str> {
    delimited(
        line_starter("-"),
        till_newline1.map(str::trim),
        Parser::or(line_ending, eof),
    )(lines)
}

fn line(lines: &str) -> IResult<&str, &str> {
    verify(
        terminated(till_newline1.map(str::trim), Parser::or(line_ending, eof)),
        |s: &str| !s.is_empty(),
    )(lines)
}

fn wait(lines: &str) -> IResult<&str, f64> {
    command_line(preceded(tuple((tag("wait"), multispace0)), double))(lines)
}

fn start_show(lines: &str) -> IResult<&str, &str> {
    command_line(tag("start"))(lines)
}

fn on_show_end(lines: &str) -> IResult<&str, &str> {
    command_line(tag("ended"))(lines)
}

fn empty(lines: &str) -> IResult<&str, &str> {
    verify(terminated(till_newline0, line_ending), |s: &str| {
        s.trim().is_empty()
    })(lines)
}

/// Skips while the first parser has input. Kinda like preceded but runs the first parser until it fails.
/// Maybe this exists in nom already but i didn't find something equivalent.
fn skip_while<I, O1, O2, F, S, E>(to_skip: S, parser: F) -> impl FnMut(I) -> IResult<I, O1, E>
where
    I: Clone + InputLength,
    F: Parser<I, O1, E>,
    S: Parser<I, O2, E>,
    E: ParseError<I>,
{
    preceded(fold_many0(to_skip, || (), |_, _| ()), parser)
}

fn prompt_block(lines: &str) -> IResult<&str, ScriptEntry> {
    let (lines, (prompt, answer_block)) = (tuple((
        skip_while(empty, prompt),
        alt((
            tuple((skip_while(empty, answer), peek(skip_while(empty, prompt))))
                .map(|(answer, _)| AnswerBlock::Single(answer.into())),
            many0(
                tuple((skip_while(empty, answer), skip_while(empty, line))).map(
                    |(answer, line)| Answer {
                        answer: answer.into(),
                        response: line.into(),
                    },
                ),
            )
            .map(AnswerBlock::Many),
        )),
    )))(lines)?;
    Ok((
        lines,
        ScriptEntry::Prompt {
            prompt: prompt.into(),
            choices: answer_block,
        },
    ))
}

fn line_block(lines: &str) -> IResult<&str, ScriptEntry> {
    skip_while(empty, line)(lines).map(|(input, line)| (input, ScriptEntry::Line(line.into())))
}

fn wait_block(lines: &str) -> IResult<&str, ScriptEntry> {
    skip_while(empty, wait)(lines)
        .map(|(input, wait)| (input, ScriptEntry::Wait(Duration::from_secs_f64(wait))))
}

fn start_show_block(lines: &str) -> IResult<&str, ScriptEntry> {
    skip_while(empty, start_show)(lines).map(|(input, _)| (input, ScriptEntry::StartShow))
}

fn on_show_end_block(lines: &str) -> IResult<&str, ScriptEntry> {
    skip_while(empty, on_show_end)(lines).map(|(input, _)| (input, ScriptEntry::OnShowEnd))
}

fn parse_entries(lines: &str) -> IResult<&str, Vec<ScriptEntry>> {
    many0(alt((
        prompt_block,
        wait_block,
        start_show_block,
        on_show_end_block,
        line_block,
    )))(lines)
}

fn parse_script(lines: &str) -> IResult<&str, ScriptEntries> {
    let (extra_input, mut entries) = parse_entries(lines)?;
    let end_start = entries
        .iter()
        .position(|entry| matches!(entry, ScriptEntry::OnShowEnd))
        .unwrap_or(entries.len());
    let after_end = entries.split_off(end_start);
    Ok((
        extra_input,
        ScriptEntries {
            before_end: entries,
            after_show_end: after_end,
        },
    ))
}

#[derive(Debug, Default, TypePath, PartialEq, Eq)]
struct ScriptEntries {
    before_end: Vec<ScriptEntry>,
    after_show_end: Vec<ScriptEntry>,
}

impl ScriptEntries {
    fn from_before_end(entries: impl Into<Vec<ScriptEntry>>) -> Self {
        ScriptEntries {
            before_end: entries.into(),
            after_show_end: Vec::new(),
        }
    }
}

#[derive(Debug, Asset, TypePath, PartialEq, Eq)]
pub struct Script {
    raw: Cow<'static, str>,
    entries: ScriptEntries,
}

impl Script {
    pub fn from_raw(
        raw: impl Into<Cow<'static, str>>,
    ) -> Result<Self, nom::Err<nom::error::Error<String>>> {
        let raw: Cow<'static, str> = raw.into();
        let (_extra_input, entries) = parse_script(&raw).map_err(|e| e.map_input(|s| s.into()))?;
        Ok(Script { raw, entries })
    }

    pub fn get_entry(&self, index: usize, after_show_end: bool) -> Option<&ScriptEntry> {
        if after_show_end {
            self.entries.after_show_end.get(index)
        } else {
            self.entries.before_end.get(index)
        }
    }
}

#[derive(Default)]
struct ScriptLoader;

#[derive(Debug)]
enum ScriptLoadError {
    Io(std::io::Error),
    Utf8(std::string::FromUtf8Error),
    Parse(nom::Err<nom::error::Error<String>>),
}

impl Display for ScriptLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => e.fmt(f),
            Self::Utf8(e) => e.fmt(f),
            Self::Parse(e) => e.fmt(f),
        }
    }
}

impl Error for ScriptLoadError {
    fn cause(&self) -> Option<&dyn Error> {
        match self {
            Self::Io(e) => Some(e),
            Self::Utf8(e) => Some(e),
            Self::Parse(e) => Some(e),
        }
    }
}

impl AssetLoader for ScriptLoader {
    type Asset = Script;
    type Error = ScriptLoadError;
    type Settings = ();
    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &Self::Settings,
        _load_context: &mut bevy::asset::LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        let mut buf = Vec::new();
        reader
            .read_to_end(&mut buf)
            .await
            .map_err(ScriptLoadError::Io)?;
        Script::from_raw(String::from_utf8(buf).map_err(ScriptLoadError::Utf8)?)
            .map_err(ScriptLoadError::Parse)
    }
    fn extensions(&self) -> &[&str] {
        &["txt"]
    }
}

/// triggered on a [`ScriptRunner`] to update its state in some way
#[derive(Clone, Debug, PartialEq, Eq, Hash, Event, Reflect)]
pub enum UpdateRunner {
    /// immediately finish displaying the current text
    FinishLine,
    /// immediately progress to the next line (ignored if currently choosing an answer)
    NextLine,
    /// if the index is invalid or the runner isn't currently choosing an answer this is ignored
    ChooseAnswer(usize),
    ShowEnded,
}

/// Triggered (on a [`ScriptRunner`] unless explicitly stated otherwise) to indicate certain state changes
#[derive(Clone, Debug, PartialEq, Eq, Hash, Event, Reflect)]
pub enum RunnerUpdated {
    HideText,
    ShowText,
    HideChoices,
    ShowChoices,
    FinishedMain,
    FinishedEnd,
    NoScript,
    /// Triggered both on the runner and globally whenever a [`ScriptEntry::StartShow`] is reached
    StartShow,
}

// a nicer version of this would have runners/choices automatically add/remove each other with
// component hooks but that's not insignificant effort and i need to get this done
#[derive(Debug, Component)]
#[require(Text)]
#[component(on_add = ScriptRunner::on_add)]
pub struct ScriptRunner {
    script: Handle<Script>,
    choices_display: Entity,
    /// If the show has ended
    show_ended: bool,
    /// If the script runner has started using the script entries after the show has ended. This
    /// should always become true after `show_ended` is true and a new entry has been reached.
    using_ended_entries: bool,
    /// Index of the current ScriptEntry
    current_entry: usize,
    /// If `current_entry` refers to a `Prompt` with `choices = AnswerBlock::Many(choice_vec)`,
    /// this is `Some(i)` where `choice_vec[i].response` is the text currently being displayed.
    /// Otherwise, this should be None.
    current_answer: Option<usize>,
    /// When `current_entry` refers to a `ScriptEntry::Wait`, this is used to time the wait.
    wait_timer: Option<Timer>,
    /// Number of bytes displayed so far
    displayed_bytes: usize,
    display_timer: Timer,
    /// Time since the current line started displaying in timer ticks
    display_ticks: u32,
    /// The timestamp of the last displayed char in timer ticks
    last_displayed_tick: u32,
    finished_line: bool,
    showing_text: bool,
    /// Whether the current set of entries (main entries or end entries) is finished
    finished_section: bool,
}

impl ScriptRunner {
    pub fn new(script: Handle<Script>, choices_display: Entity, text_speed: f32) -> Self {
        ScriptRunner {
            script,
            choices_display,
            show_ended: false,
            using_ended_entries: false,
            current_entry: 0,
            current_answer: None,
            wait_timer: None,
            displayed_bytes: 0,
            display_timer: Timer::new(
                Duration::from_secs_f32(1. / text_speed),
                TimerMode::Repeating,
            ),
            display_ticks: 0,
            last_displayed_tick: 0,
            finished_line: false,
            showing_text: false,
            finished_section: false,
        }
    }
    pub fn pause(&mut self) {
        self.display_timer.pause();
        if let Some(wait_timer) = self.wait_timer.as_mut() {
            wait_timer.pause();
        }
    }
    pub fn unpause(&mut self) {
        self.display_timer.unpause();
        if let Some(wait_timer) = self.wait_timer.as_mut() {
            wait_timer.unpause();
        }
    }
    pub fn paused(&self) -> bool {
        self.display_timer.paused()
    }
    pub fn is_line_finished(&self) -> bool {
        self.finished_line
    }
    pub fn reset(&mut self) {
        self.finished_section = false;
        self.show_ended = false;
        self.using_ended_entries = false;
        self.wait_timer = None;
        self.current_entry = 0;
        self.current_answer = None;
        self.reset_display();
    }

    fn reset_display(&mut self) {
        self.display_timer.reset();
        self.displayed_bytes = 0;
        self.display_ticks = 0;
        self.last_displayed_tick = 0;
        self.finished_line = false;
    }
    fn finish_line(&mut self, line: &str) {
        self.finished_line = true;
        self.displayed_bytes = line.len();
    }
    fn next_entry(&mut self) {
        if self.show_ended && !self.using_ended_entries {
            self.reset();
            self.show_ended = true;
            self.using_ended_entries = true;
        } else {
            self.current_entry += 1;
        }
    }

    fn on_add(mut deferred_world: DeferredWorld, entity: Entity, _: ComponentId) {
        deferred_world
            .commands()
            .entity(entity)
            .observe(ScriptRunner::on_update_runner_trigger)
            .observe(ScriptRunner::on_runner_updated);
    }
    fn on_update_runner_trigger(
        trigger: Trigger<UpdateRunner>,
        mut commands: Commands,
        mut runners: Query<(&mut ScriptRunner, &mut Text)>,
        scripts: Res<Assets<Script>>,
    ) {
        let Ok((mut runner, mut text)) = runners.get_mut(trigger.entity()) else {
            return;
        };
        let runner = &mut *runner;
        let mut runner_commands = commands.entity(trigger.entity());
        let Some(script) = scripts.get(runner.script.id()) else {
            runner_commands.trigger(RunnerUpdated::NoScript);
            return;
        };
        // this is handled early so that the runner will ALWAYS be told if the show ends, assuming
        // it has a valid script. if this isn't handled early, a runner that finished the main entries
        // won't be told the show has ended
        if *trigger.event() == UpdateRunner::ShowEnded {
            runner.show_ended = true;
        }
        let Some(entry) = script.get_entry(runner.current_entry, runner.using_ended_entries) else {
            return;
        };
        match trigger.event() {
            UpdateRunner::FinishLine => match (entry, runner.current_answer) {
                (ScriptEntry::Line(s), None) | (ScriptEntry::Prompt { prompt: s, .. }, None) => {
                    text.0.clone_from(s);
                    runner.finish_line(&text);
                    if matches!(entry, ScriptEntry::Prompt { .. }) {
                        commands
                            .trigger_targets(RunnerUpdated::ShowChoices, runner.choices_display);
                    }
                }
                (ScriptEntry::Prompt { choices, .. }, Some(index)) => {
                    let Some(response) = choices.get_response(index) else {
                        return;
                    };
                    text.0.clone_from(response);
                    runner.finish_line(&text);
                }
                (ScriptEntry::Wait(_), _)
                | (ScriptEntry::StartShow, _)
                | (ScriptEntry::OnShowEnd, _) => {}
                (ScriptEntry::Line(_), Some(_)) => unreachable!(),
            },
            UpdateRunner::NextLine => match (entry, runner.current_answer) {
                (ScriptEntry::Wait(_), _)
                | (ScriptEntry::StartShow, _)
                | (ScriptEntry::OnShowEnd, _) => {}
                (ScriptEntry::Line(_), None) => {
                    text.0.clear();
                    runner.reset_display();
                    runner.next_entry();
                }
                (ScriptEntry::Prompt { .. }, None) => {}
                (
                    ScriptEntry::Prompt {
                        choices: AnswerBlock::Many(answers),
                        ..
                    },
                    Some(current_answer),
                ) => {
                    runner.current_answer = None;
                    text.0.clear();
                    runner.reset_display();
                    runner.next_entry();
                }
                (ScriptEntry::Line(_), Some(_))
                | (
                    ScriptEntry::Prompt {
                        choices: AnswerBlock::Single(_),
                        ..
                    },
                    Some(_),
                ) => unreachable!(),
            },
            &UpdateRunner::ChooseAnswer(index) => match (entry, runner.current_answer) {
                (_, Some(_))
                | (ScriptEntry::Line(_), _)
                | (ScriptEntry::StartShow, _)
                | (ScriptEntry::Wait(_), _)
                | (ScriptEntry::OnShowEnd, _) => {}
                (
                    ScriptEntry::Prompt {
                        choices: AnswerBlock::Single(_),
                        ..
                    },
                    None,
                ) => {
                    if index == 0 {
                        text.0.clear();
                        runner.reset_display();
                        commands
                            .trigger_targets(RunnerUpdated::HideChoices, runner.choices_display);
                        runner.next_entry();
                    }
                }
                (
                    ScriptEntry::Prompt {
                        choices: AnswerBlock::Many(answers),
                        ..
                    },
                    None,
                ) => {
                    if index >= answers.len() {
                        return;
                    };
                    runner.current_answer = Some(index);
                    text.0.clear();
                    runner.reset_display();
                    commands.trigger_targets(RunnerUpdated::HideChoices, runner.choices_display);
                }
            },
            // handled above
            UpdateRunner::ShowEnded => {}
        }
    }
    fn on_runner_updated(
        trigger: Trigger<RunnerUpdated>,
        mut commands: Commands,
        mut script_runners: Query<&mut ScriptRunner>,
    ) {
        if matches!(
            *trigger.event(),
            RunnerUpdated::FinishedMain | RunnerUpdated::FinishedEnd
        ) {
            if let Ok(mut runner) = script_runners.get_mut(trigger.entity()) {
                if runner.showing_text {
                    commands
                        .entity(trigger.entity())
                        .trigger(RunnerUpdated::HideText);
                    runner.showing_text = false;
                }
            }
        }
    }
}

#[derive(Component)]
#[require(Node)]
#[component(on_add = ScriptChoices::on_add)]
pub struct ScriptChoices {
    runner: Entity,
    choice_display_root: Entity,
    displaying_choices: bool,
}

impl ScriptChoices {
    pub fn new(runner: Entity, choice_display_root: Entity) -> Self {
        ScriptChoices {
            runner,
            choice_display_root,
            displaying_choices: false,
        }
    }

    fn on_add(mut deferred_world: DeferredWorld, entity: Entity, _: ComponentId) {
        let choices = deferred_world
            .entity(entity)
            .get::<ScriptChoices>()
            .unwrap();
        let visibility = if choices.displaying_choices {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        let choice_display_root = choices.choice_display_root;
        let mut commands = deferred_world.commands();
        commands
            .entity(entity)
            .observe(ScriptChoices::on_runner_updated);
        commands.entity(choice_display_root).insert(visibility);
    }

    fn on_runner_updated(
        trigger: Trigger<RunnerUpdated>,
        mut commands: Commands,
        runners: Query<&ScriptRunner>,
        mut choices_displays: Query<(&mut ScriptChoices, Option<&Children>)>,
        mut with_visibility: Query<&mut Visibility, Without<ScriptChoices>>,
        choices: Query<&ScriptChoice>,
        scripts: Res<Assets<Script>>,
    ) {
        match trigger.event() {
            RunnerUpdated::HideText
            | RunnerUpdated::HideChoices
            | RunnerUpdated::FinishedMain
            | RunnerUpdated::FinishedEnd
            | RunnerUpdated::NoScript => {
                if let Some(mut choices_commands) = commands.get_entity(trigger.entity()) {
                    if let Ok((mut choices_display, children)) =
                        choices_displays.get_mut(trigger.entity())
                    {
                        if !choices_display.displaying_choices {
                            return;
                        }
                        choices_display.displaying_choices = false;
                        for child in children
                            .map(|children| children.deref())
                            .unwrap_or(&[])
                            .iter()
                            .filter(|&&child| choices.contains(child))
                            .copied()
                        {
                            commands.entity(child).despawn_recursive();
                        }
                        if let Ok(mut root_visibility) =
                            with_visibility.get_mut(choices_display.choice_display_root)
                        {
                            *root_visibility = Visibility::Hidden;
                        }
                    }
                }
            }
            RunnerUpdated::ShowChoices | RunnerUpdated::ShowText => {
                let (Some(mut choices_commands), Ok((mut choices_display, _))) = (
                    commands.get_entity(trigger.entity()),
                    choices_displays.get_mut(trigger.entity()),
                ) else {
                    return;
                };
                if choices_display.displaying_choices {
                    return;
                }
                choices_display.displaying_choices = true;
                let Ok(runner) = runners.get(choices_display.runner) else {
                    return;
                };
                let Some(script) = scripts.get(runner.script.id()) else {
                    return;
                };
                let Some(entry) =
                    script.get_entry(runner.current_entry, runner.using_ended_entries)
                else {
                    return;
                };
                match entry {
                    ScriptEntry::Prompt { choices, .. } => {
                        choices_commands.with_children(|builder| {
                            for (i, answer) in choices.answers_iter().enumerate() {
                                builder.spawn((ScriptChoice(i), Text(answer.clone())));
                            }
                        });
                        if let Ok(mut root_visibility) =
                            with_visibility.get_mut(choices_display.choice_display_root)
                        {
                            *root_visibility = Visibility::Inherited;
                        }
                    }
                    ScriptEntry::Line(_)
                    | ScriptEntry::Wait(_)
                    | ScriptEntry::StartShow
                    | ScriptEntry::OnShowEnd => {}
                }
            }
            RunnerUpdated::StartShow => {}
        }
    }
}

/// Added to each of the UI elements that holds the text for a given possible response to a script
/// prompt. These UI elements are spawned as children of the entity with [`ScriptChoices`] and
/// (currently) not checked for whether they have any other relevant components (although a
/// [`Text`] component is initially inserted).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Component, Reflect)]
#[require(Text)]
#[component(on_add = ScriptChoice::on_add)]
pub struct ScriptChoice(usize);

impl ScriptChoice {
    fn on_add(mut deferred_world: DeferredWorld, entity: Entity, _: ComponentId) {
        deferred_world
            .commands()
            .entity(entity)
            .observe(ScriptChoice::on_click)
            .observe(ScriptChoice::on_start_hover)
            .observe(ScriptChoice::on_stop_hover);
    }
    fn on_click(
        mut trigger: Trigger<Pointer<Click>>,
        mut commands: Commands,
        choices: Query<(&ScriptChoice, &Parent)>,
        choice_displays: Query<&ScriptChoices>,
    ) {
        trigger.propagate(false);
        if let Some((i, mut runner_commands)) = choices
            .get(trigger.entity())
            .and_then(|(choice, parent)| Ok((choice.0, choice_displays.get(parent.get())?)))
            .ok()
            .and_then(|(i, choice_display)| Some((i, commands.get_entity(choice_display.runner)?)))
        {
            runner_commands.trigger(UpdateRunner::ChooseAnswer(i));
        }
    }
    fn on_start_hover(trigger: Trigger<Pointer<Over>>, mut commands: Commands) {
        if let Some(mut commands) = commands.get_entity(trigger.entity()) {
            commands.insert(TextColor(Color::linear_rgb(0.5, 0.5, 0.5)));
        }
    }
    fn on_stop_hover(trigger: Trigger<Pointer<Out>>, mut commands: Commands) {
        if let Some(mut commands) = commands.get_entity(trigger.entity()) {
            commands.insert(TextColor::default());
        }
    }
}

fn update_script_runner_text(
    mut commands: Commands,
    mut runners: Query<(Entity, &mut ScriptRunner, &mut Text)>,
    scripts: Res<Assets<Script>>,
    time: Res<Time>,
) {
    'runners: for (runner_entity, mut runner, mut text) in runners.iter_mut() {
        if runner.paused() {
            continue;
        }
        let runner = &mut *runner;
        runner.display_ticks += runner
            .display_timer
            .tick(time.delta())
            .times_finished_this_tick();

        let Some(script) = scripts.get(runner.script.id()) else {
            continue;
        };
        // this loop is only really necessary so StartShow entries don't cost any frames to process
        loop {
            let Some(entry) = script.get_entry(runner.current_entry, runner.using_ended_entries)
            else {
                if !runner.finished_section {
                    if runner.using_ended_entries {
                        commands.trigger_targets(RunnerUpdated::FinishedEnd, runner_entity);
                    } else {
                        commands.trigger_targets(RunnerUpdated::FinishedMain, runner_entity);
                    }
                    runner.finished_section = true;
                }
                // we failed to get the current entry, which means we are not displaying ANYTHING,
                // but the show has ended, so we should definitely use the ending entries
                if runner.show_ended && !runner.using_ended_entries {
                    runner.next_entry();
                    continue;
                } else if runner.showing_text {
                    // if we failed to get the current entry, either the runner's in an invalid state,
                    // or we hit the end, in both cases text should be hidden
                    commands
                        .entity(runner_entity)
                        .trigger(RunnerUpdated::HideText);
                    runner.showing_text = false;
                }
                continue 'runners;
            };
            let current_answer = runner.current_answer;
            let mut extend_text = |s: &str| {
                let mut new_bytes = 0;
                // oh no
                text.0.extend(
                    s[runner.displayed_bytes..]
                        .chars()
                        .map(|c| match c {
                            '.' | '?' | '-' | '!' => (c, 6),
                            ',' => (c, 3),
                            _ => (c, 1),
                        })
                        .take_while(|(_, ticks)| {
                            if runner.last_displayed_tick < runner.display_ticks {
                                runner.last_displayed_tick += ticks;
                                true
                            } else {
                                false
                            }
                        })
                        .map(|(c, _)| {
                            new_bytes += c.len_utf8();
                            c
                        }),
                );
                runner.displayed_bytes += new_bytes;
            };
            let text_len;
            let currently_prompt;
            match (entry, current_answer) {
                (ScriptEntry::Line(prompt), None) | (ScriptEntry::Prompt { prompt, .. }, None) => {
                    extend_text(prompt);
                    text_len = prompt.as_bytes().len();
                    currently_prompt = matches!(entry, ScriptEntry::Prompt { .. });
                    if !runner.showing_text {
                        commands
                            .entity(runner_entity)
                            .trigger(RunnerUpdated::ShowText);
                        runner.showing_text = true;
                    }
                }
                (
                    ScriptEntry::Prompt {
                        choices: AnswerBlock::Many(answers),
                        ..
                    },
                    Some(answer_index),
                ) => {
                    extend_text(&answers[answer_index].response);
                    text_len = answers[answer_index].response.as_bytes().len();
                    currently_prompt = false;
                    if !runner.showing_text {
                        commands
                            .entity(runner_entity)
                            .trigger(RunnerUpdated::ShowText);
                        runner.showing_text = true;
                    }
                }
                (ScriptEntry::Wait(wait), _) => {
                    let wait_timer = runner.wait_timer.get_or_insert_with(|| {
                        if runner.showing_text {
                            commands
                                .entity(runner_entity)
                                .trigger(RunnerUpdated::HideText);
                            runner.showing_text = false;
                        }
                        Timer::new(*wait, TimerMode::Once)
                    });
                    if wait_timer.tick(time.delta()).finished() {
                        runner.wait_timer = None;
                        runner.reset_display();
                        runner.next_entry();
                    }
                    // this shouldn't deal with text_len or currently_prompt at all
                    break;
                }
                (ScriptEntry::StartShow, _) => {
                    commands.trigger_targets(RunnerUpdated::StartShow, runner_entity);
                    runner.reset_display();
                    text.0.clear();
                    runner.next_entry();
                    continue;
                }
                (ScriptEntry::OnShowEnd, _) => {
                    runner.reset_display();
                    text.0.clear();
                    runner.next_entry();
                    continue;
                }
                (ScriptEntry::Line(_), Some(_))
                | (
                    ScriptEntry::Prompt {
                        choices: AnswerBlock::Single(_),
                        ..
                    },
                    Some(_),
                ) => unreachable!(),
            }
            if runner.displayed_bytes >= text_len && !runner.is_line_finished() {
                runner.finish_line(&text.0);
                if currently_prompt {
                    commands.trigger_targets(RunnerUpdated::ShowChoices, runner.choices_display);
                }
            }
            break;
        }
    }
}

fn on_script_reload_reset_runners(
    trigger: Trigger<AssetEvent<Script>>,
    mut commands: Commands,
    mut runners: Query<(&mut ScriptRunner, &mut Text)>,
) {
    match trigger.event() {
        AssetEvent::Added { id }
        | AssetEvent::Modified { id }
        | AssetEvent::LoadedWithDependencies { id } => {
            for (mut runner, mut text) in runners.iter_mut() {
                if runner.script.id() == *id {
                    runner.reset();
                    runner.unpause();
                    text.0.clear();
                    commands.trigger_targets(RunnerUpdated::HideChoices, runner.choices_display);
                }
            }
        }
        _ => {}
    }
}

pub struct ScriptPlugin;

impl Plugin for ScriptPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<Script>()
            .init_asset_loader::<ScriptLoader>()
            .add_event::<UpdateRunner>()
            .add_event::<RunnerUpdated>()
            .add_systems(Update, update_script_runner_text)
            .add_observer(on_script_reload_reset_runners);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_script_parses() {
        let empties = &["", "   ", "\t "];
        for &empty in empties {
            assert_eq!(
                Script::from_raw(empty),
                Ok(Script {
                    raw: Cow::Borrowed(empty),
                    entries: ScriptEntries::default(),
                })
            );
        }
    }

    #[test]
    fn single_line_parses() {
        let lines = &["test", " test", "test ", "test\n", " test \r\n"];
        for &line in lines {
            assert_eq!(
                Script::from_raw(line),
                Ok(Script {
                    raw: Cow::Borrowed(line),
                    entries: ScriptEntries::from_before_end(vec![ScriptEntry::Line(
                        line.trim().into()
                    )])
                })
            );
        }
    }

    #[test]
    fn prompt_block_parses() {
        let single_answer = ">prompt\r\n\n\n-   answer\n>next prompt\n-next 1\nnext response 1\n-next 2\nnext response 2";
        assert_eq!(
            Script::from_raw(single_answer),
            Ok(Script {
                raw: single_answer.into(),
                entries: ScriptEntries::from_before_end(vec![
                    ScriptEntry::Prompt {
                        prompt: "prompt".into(),
                        choices: AnswerBlock::Single("answer".into())
                    },
                    ScriptEntry::Prompt {
                        prompt: "next prompt".into(),
                        choices: AnswerBlock::Many(vec![
                            Answer {
                                answer: "next 1".into(),
                                response: "next response 1".into(),
                            },
                            Answer {
                                answer: "next 2".into(),
                                response: "next response 2".into(),
                            },
                        ])
                    }
                ])
            })
        );

        let multi_answer = ">prompt\n\n\r\n- answer1  \r\n response1\t\n-answer2\r\nresponse2\n";
        assert_eq!(
            Script::from_raw(multi_answer),
            Ok(Script {
                raw: multi_answer.into(),
                entries: ScriptEntries::from_before_end(vec![ScriptEntry::Prompt {
                    prompt: "prompt".into(),
                    choices: AnswerBlock::Many(vec![
                        Answer {
                            answer: "answer1".into(),
                            response: "response1".into(),
                        },
                        Answer {
                            answer: "answer2".into(),
                            response: "response2".into(),
                        },
                    ])
                }])
            })
        );
    }

    #[test]
    fn wait_blocks_parse() {
        let to_parse = " \n [ wait 1.0 ]\r\n[wait 0.25]\n [wait 10]\t \r\n";
        assert_eq!(
            Script::from_raw(to_parse),
            Ok(Script {
                raw: to_parse.into(),
                entries: ScriptEntries::from_before_end(vec![
                    ScriptEntry::Wait(Duration::from_secs_f64(1.)),
                    ScriptEntry::Wait(Duration::from_secs_f64(0.25)),
                    ScriptEntry::Wait(Duration::from_secs_f64(10.)),
                ])
            })
        );
    }

    #[test]
    fn start_show_blocks_parse() {
        let to_parse = "  [ start ]\r\n[start  ]\n [ start ] \t \r\n";
        assert_eq!(
            Script::from_raw(to_parse),
            Ok(Script {
                raw: to_parse.into(),
                entries: ScriptEntries::from_before_end(vec![
                    ScriptEntry::StartShow,
                    ScriptEntry::StartShow,
                    ScriptEntry::StartShow,
                ])
            })
        );
    }

    #[test]
    fn on_show_end_blocks_parse() {
        let to_parse = "  [ ended ]\r\n[ended  ]\n [ ended ] \t \r\n";
        assert_eq!(
            Script::from_raw(to_parse),
            Ok(Script {
                raw: to_parse.into(),
                entries: ScriptEntries {
                    before_end: vec![],
                    after_show_end: vec![
                        ScriptEntry::OnShowEnd,
                        ScriptEntry::OnShowEnd,
                        ScriptEntry::OnShowEnd
                    ]
                }
            })
        );
    }

    #[test]
    fn mixed_blocks_parse() {
        let to_parse = "line 1\r\n\t[wait1]\t\n\n[start  \t]\n\n [   wait 120. ]  \n\n> prompt \r\n-   answer\n response  \r\n line 2\n \n [ start\t]  \r\n line 3 \r\n";
        assert_eq!(
            Script::from_raw(to_parse),
            Ok(Script {
                raw: to_parse.into(),
                entries: ScriptEntries::from_before_end(vec![
                    ScriptEntry::Line("line 1".into()),
                    ScriptEntry::Wait(Duration::from_secs_f64(1.0)),
                    ScriptEntry::StartShow,
                    ScriptEntry::Wait(Duration::from_secs_f64(120.0)),
                    ScriptEntry::Prompt {
                        prompt: "prompt".into(),
                        choices: AnswerBlock::Many(vec![Answer {
                            answer: "answer".into(),
                            response: "response".into(),
                        }])
                    },
                    ScriptEntry::Line("line 2".into()),
                    ScriptEntry::StartShow,
                    ScriptEntry::Line("line 3".into()),
                ])
            })
        );
    }
}
