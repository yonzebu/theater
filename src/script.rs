use std::{borrow::Cow, error::Error, fmt::Display, time::Duration};

use bevy::{
    asset::{io::Reader, AssetLoader}, ecs::component::StorageType, prelude::*, time::Stopwatch
};
use nom::{
    branch::alt,
    bytes::complete::{tag, take_till, take_till1},
    character::complete::{line_ending, multispace0},
    combinator::{eof, peek, rest, verify},
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
    Single(String),
    Multi(Vec<Answer>),
}

impl AnswerBlock {
    pub fn get_answer(&self, answer: usize) -> Option<&String> {
        match self {
            Self::Single(s) => (answer == 0).then(|| s),
            Self::Multi(choices) => choices.get(answer).map(|answer| &answer.answer)
        }
    }
    pub fn get_response(&self, answer: usize) -> Option<&String> {
        match self {
            Self::Single(s) => None,
            Self::Multi(choices) => choices.get(answer).map(|answer| &answer.response)
        }
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

fn prompt<'a>(lines: &'a str) -> IResult<&'a str, &'a str> {
    delimited(
        line_starter(">"),
        till_newline1.map(str::trim),
        Parser::or(line_ending, eof),
    )(lines)
}

fn answer<'a>(lines: &'a str) -> IResult<&'a str, &'a str> {
    delimited(
        line_starter("-"),
        till_newline1.map(str::trim),
        Parser::or(line_ending, eof),
    )(lines)
}

fn end<'a>(lines: &'a str) -> IResult<&'a str, &'a str> {
    delimited(
        line_starter("!"),
        till_newline1.map(str::trim),
        Parser::or(line_ending, eof),
    )(lines)
}

fn line<'a>(lines: &'a str) -> IResult<&'a str, &'a str> {
    verify(terminated(till_newline1.map(str::trim), Parser::or(line_ending, eof)), |s: &str| s.len() > 0)(lines)
}

fn empty<'a>(lines: &'a str) -> IResult<&'a str, &'a str> {
    verify(terminated(till_newline0, line_ending), |s: &str| s.trim().len() == 0)(lines)
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

fn prompt_block<'a>(lines: &'a str) -> IResult<&'a str, ScriptEntry> {
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
            .map(AnswerBlock::Multi),
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

fn line_block<'a>(lines: &'a str) -> IResult<&'a str, ScriptEntry> {
    skip_while(empty, line)(lines).map(|(input, line)| (input, ScriptEntry::Line(line.into())))
}

fn parse_entries<'a>(lines: &'a str) -> IResult<&'a str, Vec<ScriptEntry>> {
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
        let (extra_input, entries) = parse_entries(&raw).map_err(|e| e.map_input(|s| s.into()))?;
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
        settings: &Self::Settings,
        load_context: &mut bevy::asset::LoadContext<'_>,
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

#[derive(Debug)]
pub struct TextSpeed {
    pub fast: f32,
    pub slow: f32,
    pub is_fast: bool,
}

impl TextSpeed {
    pub fn new(fast: f32, slow: f32) -> Self {
        TextSpeed { fast, slow, is_fast: false }
    }
}

pub trait ScriptSpeed {
    fn speed_up(&mut self) {}
    fn slow_down(&mut self) {}
    fn char_delay(&self, remaining: &str) -> Duration;
    fn char_delay_batch(&self, remaining: &str) -> (Duration, usize) {
        (self.char_delay(remaining), 1)
    }
}
impl ScriptSpeed for f32 {
    fn char_delay(&self, _remaining: &str) -> Duration {
        Duration::from_secs_f32(*self)
    }
    fn char_delay_batch(&self, remaining: &str) -> (Duration, usize) {
        (self.char_delay(remaining), remaining.chars().count())
    }
}
impl ScriptSpeed for f64 {
    fn char_delay(&self, _remaining: &str) -> Duration {
        Duration::from_secs_f64(*self)
    }
    fn char_delay_batch(&self, remaining: &str) -> (Duration, usize) {
        (self.char_delay(remaining), remaining.chars().count())
    }
}
impl ScriptSpeed for Duration {
    fn char_delay(&self, _remaining: &str) -> Duration {
        *self
    }
    fn char_delay_batch(&self, remaining: &str) -> (Duration, usize) {
        (self.char_delay(remaining), remaining.chars().count())
    }
}
impl<F: Fn(char) -> S, S: ScriptSpeed> ScriptSpeed for F {
    fn char_delay(&self, remaining: &str) -> Duration {
        self(remaining.chars().nth(0).unwrap()).char_delay(remaining)
    }
    fn char_delay_batch(&self, remaining: &str) -> (Duration, usize) {
        let current_delay = self(remaining.chars().nth(0).unwrap()).char_delay(remaining);
        (
            current_delay, 
            remaining.chars().take_while(|&c| self(c).char_delay(remaining) == current_delay).count()
        )
    }
}
impl ScriptSpeed for TextSpeed {
    fn speed_up(&mut self) {
        self.is_fast = true;
    }
    fn slow_down(&mut self) {
        self.is_fast = false;
    }
    fn char_delay(&self, _remaining: &str) -> Duration {
        if self.is_fast {
            Duration::from_secs_f32(1. / self.fast)
        } else {
            Duration::from_secs_f32(1. / self.slow)
        }
    }
}

pub struct ScriptRunner {
    pub script: Handle<Script>,
    pub text_speed: Box<dyn ScriptSpeed + Send + Sync + 'static>,
    pub choice_displays: Vec<Entity>,
    current_entry: usize,
    current_choice: Option<usize>,
    displayed_chars: usize,
    display_timer: Stopwatch,
}

impl Component for ScriptRunner {
    const STORAGE_TYPE: bevy::ecs::component::StorageType = StorageType::Table;
    fn register_component_hooks(hooks: &mut bevy::ecs::component::ComponentHooks) {
        hooks.on_add(|mut deferred_world, entity, _| {
            deferred_world
                .commands()
                .entity(entity)
                .observe(ScriptRunner::on_update_system);
        });
    }
}

impl ScriptRunner {
    pub fn new(
        script: impl Into<Handle<Script>>, 
        text_speed: impl ScriptSpeed + Send + Sync + 'static,
        choices: impl IntoIterator<Item = Entity>
    ) -> Self {
        ScriptRunner {
            script: script.into(),
            text_speed: Box::new(text_speed),
            choice_displays: choices.into_iter().collect(),
            current_entry: 0,
            current_choice: None,
            displayed_chars: 0,
            display_timer: Stopwatch::new(),
        }
    }

    fn reset_display(&mut self) {
        self.displayed_chars = 0;
        self.display_timer.reset();
    }

    fn on_update_system(
        trigger: Trigger<UpdateRunner>, 
        mut commands: Commands,
        mut runners: Query<(&mut ScriptRunner, &mut Text)>, 
        scripts: Res<Assets<Script>>,
    ) {
        let Ok((mut runner, mut text)) = runners.get_mut(trigger.entity()) else {
            return;
        };
        let mut entity_commands = commands.entity(trigger.entity());
        match trigger.event() {
            UpdateRunner::SpeedUp => runner.text_speed.speed_up(),
            UpdateRunner::SlowDown => runner.text_speed.slow_down(),
            UpdateRunner::FinishLine => {
                let Some(script) = scripts.get(runner.script.id()) else {
                    entity_commands.trigger(RunnerUpdated::NoScript);
                    return;
                };
                let Some(entry) = script.get_entry(runner.current_entry) else {
                    entity_commands.trigger(RunnerUpdated::Finished);
                    return;
                };
                let mut display_choices = false;
                match entry {
                    ScriptEntry::Line(line) => text.0.clone_from(line),
                    ScriptEntry::Prompt { prompt, choices } => {
                        match runner.current_choice {
                            Some(choice) => {
                                let Some(response) = choices.get_response(choice) else {
                                    entity_commands.trigger(RunnerUpdated::InvalidState);
                                    return;
                                };
                                text.0.clone_from(response);
                            },
                            None => {
                                text.0.clone_from(prompt);
                                display_choices = true;
                            }
                        }
                    }
                }
                entity_commands.trigger(RunnerUpdated::FinishedLine);
                if display_choices {
                    entity_commands.trigger(RunnerUpdated::ShowChoices { script_entry: runner.current_entry });
                }
            }
            UpdateRunner::NextLine => {
                todo!();
                runner.current_entry += 1;
                text.0.clear();
                runner.reset_display();
                entity_commands.trigger(RunnerUpdated::StartedLine);
            }
            UpdateRunner::ChooseAnswer { answer } => {
                todo!();
                let Some(script) = scripts.get(runner.script.id()) else {
                    entity_commands.trigger(RunnerUpdated::NoScript);
                    return;
                };
                runner.current_entry += 1;
                runner.current_choice = Some(*answer);
                text.0.clear();
                entity_commands.trigger(RunnerUpdated::HideChoices);
                entity_commands.trigger(RunnerUpdated::FinishedLine);
                let old_entry = runner.current_entry;
                match script.get_entry(old_entry) {
                    Some(ScriptEntry::Prompt { choices, .. }) => {

                    }
                    Some(_) => {
                        entity_commands.trigger(RunnerUpdated::InvalidState);
                        return;
                    }
                    None => {
                        entity_commands.trigger(RunnerUpdated::Finished);
                        return;
                    }
                }
            }
        }
    }
}

#[derive(Component)]
pub struct ScriptChoices {
    pub runner: Entity
}

#[derive(Component)]
struct ScriptChoice;

#[derive(Debug, Clone, Event)]
pub enum RunnerUpdated {
    /// choices are currently needed and should be shown
    ShowChoices {
        script_entry: usize,
    },
    /// choices are no longer needed and should be hidden
    HideChoices,
    /// The current line has finished displaying
    FinishedLine,
    /// the next line has started displaying
    StartedLine,
    /// a new character has been displayed
    NewChar(char),
    /// the script is finished
    Finished,
    /// unable to find the script specified by the runner
    NoScript,
    /// any invalid state was reached
    InvalidState,
}

#[derive(Debug, Clone, Event)]
pub enum UpdateRunner {
    ChooseAnswer {
        answer: usize,
    },
    SpeedUp,
    SlowDown,
    /// display all of the current line, then stop updating it
    FinishLine,
    /// display the next line
    NextLine
}

fn update_script_runners(
    mut commands: Commands,
    runners: Query<(Entity, &ScriptRunner, &mut Text)>,
    choice_containers: Query<(Entity, &ScriptChoices, &Node, &Children)>,
    choices: Query<&Text, With<ScriptChoice>>,
    scripts: Res<Assets<Script>>,
) {

}

pub struct ScriptPlugin;

impl Plugin for ScriptPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<Script>()
            .init_asset_loader::<ScriptLoader>();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_script_parses() {
        let empties = &[
            "",
            "   ",
            "\t ",
        ];
        for &empty in empties {
            assert_eq!(Script::from_raw(empty), Ok(Script { raw: Cow::Borrowed(empty), entries: vec![] }));
        }
    }

    #[test]
    fn single_line_parses() {
        let lines = &[
            "test",
            " test",
            "test ",
            "test\n",
            " test \r\n"
        ];
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
                        choices: AnswerBlock::Multi(vec![
                            Answer { answer: "next 1".into(), response: "next response 1".into(), is_end: false },
                            Answer { answer: "next 2".into(), response: "next response 2".into(), is_end: false },
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
                    choices: AnswerBlock::Multi(vec![
                        Answer { answer: "answer1".into(), response: "response1".into(), is_end: false },
                        Answer { answer: "answer2".into(), response: "response2".into(), is_end: false },
                        Answer { answer: "end answer".into(), response: "end response".into(), is_end: true },
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
                        choices: AnswerBlock::Multi(vec![Answer { 
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