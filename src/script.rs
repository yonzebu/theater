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
    sequence::{delimited, preceded, terminated, tuple},
    IResult, InputLength, Parser,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Answer {
    pub answer: String,
    pub response: String,
    pub is_end: bool,
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
}

fn line_starter<'a>(starter: &'a str) -> impl FnMut(&'a str) -> IResult<&'a str, &'a str> {
    preceded(multispace0, tag(starter))
}

fn till_newline0(input: &str) -> IResult<&str, &str> {
    take_till(|c| c == '\r' || c == '\n')(input)
}

fn till_newline1(input: &str) -> IResult<&str, &str> {
    take_till1(|c| c == '\r' || c == '\n')(input)
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

fn end(lines: &str) -> IResult<&str, &str> {
    delimited(
        line_starter("!"),
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

fn empty(lines: &str) -> IResult<&str, &str> {
    verify(terminated(till_newline0, line_ending), |s: &str| {
        s.trim().is_empty()
    })(lines)
}

/// skips while the first parser has input. kinda like preceded but runs the first parser until it fails.
/// maybe this exists in nom already but i didn't find something equivalent
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
            many0(alt((
                tuple((skip_while(empty, answer), skip_while(empty, end))).map(|(answer, line)| {
                    Answer {
                        answer: answer.into(),
                        response: line.into(),
                        is_end: true,
                    }
                }),
                tuple((skip_while(empty, answer), skip_while(empty, line))).map(
                    |(answer, line)| Answer {
                        answer: answer.into(),
                        response: line.into(),
                        is_end: false,
                    },
                ),
            )))
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

fn parse_entries(lines: &str) -> IResult<&str, Vec<ScriptEntry>> {
    many0(alt((prompt_block, line_block)))(lines)
}

#[derive(Debug, Asset, TypePath, PartialEq, Eq)]
pub struct Script {
    raw: Cow<'static, str>,
    entries: Vec<ScriptEntry>,
}

impl Script {
    pub fn from_raw(
        raw: impl Into<Cow<'static, str>>,
    ) -> Result<Self, nom::Err<nom::error::Error<String>>> {
        let raw: Cow<'static, str> = raw.into();
        let (_extra_input, entries) = parse_entries(&raw).map_err(|e| e.map_input(|s| s.into()))?;
        Ok(Script { raw, entries })
    }

    pub fn get_entry(&self, index: usize) -> Option<&ScriptEntry> {
        self.entries.get(index)
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
#[derive(Debug, Event)]
pub enum UpdateRunner {
    /// immediately finish displaying the current text
    FinishLine,
    /// immediately progress to the next line (ignored if currently choosing an answer)
    NextLine,
    /// if the index is invalid or the runner isn't currently choosing an answer this is ignored
    ChooseAnswer(usize),
}

/// triggered on a [`ScriptRunner`] to indicate certain state changes
#[derive(Debug, Event)]
pub enum RunnerUpdated {
    HideChoices,
    ShowChoices,
    Finished,
    NoScript,
}

// a nicer version of this would have runners/choices automatically add/remove each other with
// component hooks but that's not insignificant effort and i need to get this done
#[derive(Debug, Component)]
#[require(Text)]
#[component(on_add = ScriptRunner::on_add)]
pub struct ScriptRunner {
    script: Handle<Script>,
    choices_display: Entity,
    /// index of the current ScriptEntry
    current_entry: usize,
    /// if this is None, the text currently being displayed is either a single line (if
    /// `current_entry` refers to a `ScriptEntry::Line`) or a prompt (`current_entry` refers to a
    /// `ScriptEntry::Prompt`). if `current_entry` refers to a `Prompt` with
    /// `choices = AnswerBlock::Many(choice_vec)`, this is `Some(i)` where
    /// `choice_vec[i].response` is the text currently being displayed.
    current_answer: Option<usize>,
    /// number of bytes displayed so far
    displayed_bytes: usize,
    display_timer: Timer,
    /// time since the current line started displaying in timer ticks
    display_ticks: u32,
    /// the timestamp of the last displayed char in timer ticks
    last_displayed_tick: u32,
    finished_line: bool,
}

impl ScriptRunner {
    pub fn new(script: Handle<Script>, choices_display: Entity, text_speed: f32) -> Self {
        ScriptRunner {
            script,
            choices_display,
            current_entry: 0,
            current_answer: None,
            displayed_bytes: 0,
            display_timer: Timer::new(
                Duration::from_secs_f32(1. / text_speed),
                TimerMode::Repeating,
            ),
            display_ticks: 0,
            last_displayed_tick: 0,
            finished_line: false,
        }
    }
    pub fn pause(&mut self) {
        self.display_timer.pause();
    }
    pub fn unpause(&mut self) {
        self.display_timer.unpause();
    }
    pub fn paused(&self) -> bool {
        self.display_timer.paused()
    }
    pub fn is_line_finished(&self) -> bool {
        self.finished_line
    }
    pub fn reset(&mut self) {
        self.current_entry = 0;
        self.current_answer = None;
        self.displayed_bytes = 0;
        self.display_timer.reset();
        self.display_ticks = 0;
        self.last_displayed_tick = 0;
        self.finished_line = false;
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

    fn on_add(mut deferred_world: DeferredWorld, entity: Entity, _: ComponentId) {
        deferred_world
            .commands()
            .entity(entity)
            .observe(ScriptRunner::on_update_runner_trigger);
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
        let Some(entry) = script.get_entry(runner.current_entry) else {
            runner_commands.trigger(RunnerUpdated::Finished);
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
                    commands.trigger_targets(RunnerUpdated::ShowChoices, runner.choices_display);
                }
                _ => unreachable!(),
            },
            UpdateRunner::NextLine => match (entry, runner.current_answer) {
                (ScriptEntry::Line(_), None) => {
                    runner.current_entry += 1;
                    text.0.clear();
                    runner.reset_display();
                }
                (ScriptEntry::Prompt { .. }, None) => {}
                (
                    ScriptEntry::Prompt {
                        choices: AnswerBlock::Many(answers),
                        ..
                    },
                    Some(current_answer),
                ) => {
                    if answers
                        .get(current_answer)
                        .map_or(false, |answer| answer.is_end)
                    {
                        runner.current_entry = usize::MAX;
                        runner.current_answer = None;
                        text.0.clear();
                        runner.reset_display();
                        runner_commands.trigger(RunnerUpdated::Finished);
                    } else {
                        runner.current_entry += 1;
                        runner.current_answer = None;
                        text.0.clear();
                        runner.reset_display();
                    }
                }
                _ => unreachable!(),
            },
            &UpdateRunner::ChooseAnswer(index) => match (entry, runner.current_answer) {
                (_, Some(_)) | (ScriptEntry::Line(_), _) => {}
                (
                    ScriptEntry::Prompt {
                        choices: AnswerBlock::Single(_),
                        ..
                    },
                    None,
                ) => {
                    if index == 0 {
                        runner.current_entry += 1;
                        text.0.clear();
                        runner.reset_display();
                        commands
                            .trigger_targets(RunnerUpdated::HideChoices, runner.choices_display);
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
            RunnerUpdated::HideChoices | RunnerUpdated::Finished | RunnerUpdated::NoScript => {
                if let Some(mut choices_commands) = commands.get_entity(trigger.entity()) {
                    if let Ok((mut choices_display, children)) =
                        choices_displays.get_mut(trigger.entity())
                    {
                        if !choices_display.displaying_choices {
                            return;
                        }
                        choices_display.displaying_choices = false;
                        let to_remove = children
                            .map(|children| children.deref())
                            .unwrap_or(&[])
                            .iter()
                            .filter(|&&child| choices.contains(child))
                            .copied()
                            .collect::<Vec<_>>();
                        choices_commands.remove_children(&to_remove);
                        if let Ok(mut root_visibility) =
                            with_visibility.get_mut(choices_display.choice_display_root)
                        {
                            *root_visibility = Visibility::Hidden;
                        }
                    }
                }
            }
            RunnerUpdated::ShowChoices => {
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
                let Some(entry) = script.get_entry(runner.current_entry) else {
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
                    ScriptEntry::Line(_) => {}
                }
            }
        }
    }
}

#[derive(Component)]
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
        trigger: Trigger<Pointer<Click>>,
        mut commands: Commands,
        choices: Query<(&ScriptChoice, &Parent)>,
        choice_displays: Query<&ScriptChoices>,
    ) {
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
            commands.insert(BackgroundColor(Color::srgba(0.867, 0.75, 0.93, 0.5)));
        }
    }
    fn on_stop_hover(trigger: Trigger<Pointer<Out>>, mut commands: Commands) {
        if let Some(mut commands) = commands.get_entity(trigger.entity()) {
            commands.insert(BackgroundColor(Color::srgba(0.8, 0.43, 1., 0.)));
        }
    }
}

fn update_script_runner_text(
    mut commands: Commands,
    mut runners: Query<(&mut ScriptRunner, &mut Text)>,
    scripts: Res<Assets<Script>>,
    time: Res<Time>,
) {
    for (mut runner, mut text) in runners.iter_mut() {
        let runner = &mut *runner;
        runner.display_ticks += runner
            .display_timer
            .tick(time.delta())
            .times_finished_this_tick();

        let Some(script) = scripts.get(runner.script.id()) else {
            continue;
        };
        let Some(entry) = script.get_entry(runner.current_entry) else {
            continue;
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
            (ScriptEntry::Line(line), None) => {
                extend_text(line);
                text_len = line.as_bytes().len();
                currently_prompt = false;
            }
            (ScriptEntry::Prompt { prompt, .. }, None) => {
                extend_text(prompt);
                text_len = prompt.as_bytes().len();
                currently_prompt = true;
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
            }
            _ => unreachable!(),
        }
        if runner.displayed_bytes >= text_len && !runner.is_line_finished() {
            runner.finish_line(&text.0);
            if currently_prompt {
                commands.trigger_targets(RunnerUpdated::ShowChoices, runner.choices_display);
            }
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
                    entries: vec![]
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
                    entries: vec![ScriptEntry::Line(line.trim().into())]
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
                entries: vec![
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
                                is_end: false
                            },
                            Answer {
                                answer: "next 2".into(),
                                response: "next response 2".into(),
                                is_end: false
                            },
                        ])
                    }
                ]
            })
        );

        let multi_answer = ">prompt\n\n\r\n- answer1  \r\n response1\t\n-answer2\r\nresponse2\n-end answer\n\r\n!end response";
        assert_eq!(
            Script::from_raw(multi_answer),
            Ok(Script {
                raw: multi_answer.into(),
                entries: vec![ScriptEntry::Prompt {
                    prompt: "prompt".into(),
                    choices: AnswerBlock::Many(vec![
                        Answer {
                            answer: "answer1".into(),
                            response: "response1".into(),
                            is_end: false
                        },
                        Answer {
                            answer: "answer2".into(),
                            response: "response2".into(),
                            is_end: false
                        },
                        Answer {
                            answer: "end answer".into(),
                            response: "end response".into(),
                            is_end: true
                        },
                    ])
                }]
            })
        );
    }

    #[test]
    fn mixed_blocks_parse() {
        let to_parse = "line 1\n> prompt \r\n-   answer\n response  \r\n line 2\n \n line 3 \r\n";
        assert_eq!(
            Script::from_raw(to_parse),
            Ok(Script {
                raw: to_parse.into(),
                entries: vec![
                    ScriptEntry::Line("line 1".into()),
                    ScriptEntry::Prompt {
                        prompt: "prompt".into(),
                        choices: AnswerBlock::Many(vec![Answer {
                            answer: "answer".into(),
                            response: "response".into(),
                            is_end: false
                        }])
                    },
                    ScriptEntry::Line("line 2".into()),
                    ScriptEntry::Line("line 3".into()),
                ]
            })
        );
    }
}
