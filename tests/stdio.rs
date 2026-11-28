//! Command (stdio) upstreams: a real child process speaking MCP over stdin/stdout.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use mcp_hub::{
    config::parse_config,
    secrets::SecretStore,
    upstream::{Caller, Timings, Upstream, UpstreamError},
};
use rmcp::model::{CallToolRequestParams, CallToolResponse};
use serde_json::json;
use tokio_util::sync::CancellationToken;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

fn upstream(server: serde_json::Value, store: &SecretStore) -> Arc<Upstream> {
    upstream_with(server, store, Timings::default())
}

fn upstream_with(
    server: serde_json::Value,
    store: &SecretStore,
    timings: Timings,
) -> Arc<Upstream> {
    let config = parse_config(&json!({"servers": {"local": server}}).to_string()).unwrap();
    Upstream::with_timings(config.servers[0].clone(), store.clone(), timings)
}

/// A file a test server appends a line to every time it starts.
struct StartLog(PathBuf);

impl StartLog {
    fn new(test: &str) -> Self {
        let path = std::env::temp_dir().join(format!("mcp-hub-{test}-{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        Self(path)
    }

    fn lines(&self) -> Vec<String> {
        std::fs::read_to_string(&self.0)
            .map(|s| s.lines().map(str::to_owned).collect())
            .unwrap_or_default()
    }

    fn starts(&self) -> usize {
        self.lines().len()
    }

    /// Whether the process that started last is still there.
    fn running(&self) -> bool {
        self.nth_running(self.starts() - 1)
    }

    /// Whether the `n`th process that started (from 0) is still there.
    fn nth_running(&self, n: usize) -> bool {
        std::process::Command::new("kill")
            .args(["-0", &self.lines()[n]])
            .stderr(std::process::Stdio::null())
            .status()
            .unwrap()
            .success()
    }
}

impl Drop for StartLog {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

async fn call_text(upstream: &Upstream, tool: &str) -> Result<String, UpstreamError> {
    let response = upstream
        .call_tool(
            CallToolRequestParams::new(tool.to_owned()),
            CancellationToken::new(),
        )
        .await?;
    let CallToolResponse::Complete(result) = response else {
        panic!("expected a complete result")
    };
    Ok(result.content[0].as_text().unwrap().text.clone())
}

#[tokio::test]
async fn runs_a_command_server() {
    let store = SecretStore::memory();
    let local = upstream(
        json!({
            "command": "/bin/sh",
            "args": [fixtures().join("stdio-server.sh")],
            "cwd": fixtures(),
            "secretEnv": ["FAKE_SECRET"],
        }),
        &store,
    );

    // The secret is not in the keyring yet: the process is not even started.
    let err = local.list_tools().await.unwrap_err();
    assert!(matches!(err, UpstreamError::NeedsLogin(_)), "{err}");
    assert!(
        err.to_string()
            .contains("mcp-hub set-token local FAKE_SECRET"),
        "{err}"
    );

    store.set("local:env:fake_secret", "s3cret").await.unwrap();
    let tools = local.list_tools().await.unwrap();
    assert_eq!(
        tools.iter().map(|t| t.name.as_ref()).collect::<Vec<_>>(),
        ["env", "crash"]
    );

    let expected = format!(
        "cwd={} secret=s3cret",
        fixtures().canonicalize().unwrap().display()
    );
    assert_eq!(call_text(&local, "env").await.unwrap(), expected);
    // It answered MCP 2026-07-28's first request with an error: an older version, then.
    assert_eq!(local.protocol().as_deref(), Some("2025-11-25"));

    // The process dies mid-call: that call fails, the next one gets a new process.
    assert!(call_text(&local, "crash").await.is_err());
    assert_eq!(call_text(&local, "env").await.unwrap(), expected);

    local.close().await;
}

/// A server that does not answer a request it does not know, as MCP 2026-07-28's first, is
/// connected to the old way after 10 seconds; after that at once, also by a restarted hub.
#[tokio::test]
async fn remembers_a_server_that_speaks_an_older_version() {
    let log = StartLog::new("older");
    let saved = std::env::temp_dir().join(format!("mcp-hub-older-lists-{}", std::process::id()));
    let server = json!({
        "command": "/bin/sh",
        "args": [fixtures().join("stdio-server.sh")],
        "env": {"IGNORE_UNKNOWN": "1", "START_LOG": log.0},
    });
    let store = SecretStore::memory();
    let first = upstream(server.clone(), &store);
    first.keep_tools_in(&saved);
    let started = std::time::Instant::now();
    assert_eq!(first.list_tools().await.unwrap().len(), 2);
    assert!(started.elapsed() >= Duration::from_secs(9));
    assert_eq!(first.protocol().as_deref(), Some("2025-11-25"));
    // One process: the old way followed on the same connection.
    assert_eq!(log.starts(), 1);
    first.close().await;

    let again = std::time::Instant::now();
    assert!(call_text(&first, "env").await.is_ok());
    assert!(again.elapsed() < Duration::from_secs(5));
    first.close().await;

    // A restarted hub, which reads what the first one saved.
    let restarted = upstream(server, &store);
    restarted.keep_tools_in(&saved);
    let again = std::time::Instant::now();
    assert!(call_text(&restarted, "env").await.is_ok());
    assert!(again.elapsed() < Duration::from_secs(5));
    restarted.close().await;
    let _ = std::fs::remove_dir_all(&saved);
}

#[tokio::test]
async fn reports_why_a_command_did_not_start() {
    let store = SecretStore::memory();
    let broken = upstream(
        json!({"command": "/bin/sh", "args": ["-c", "echo 'Error: Cannot find module tsx' >&2; exit 1"]}),
        &store,
    );
    let err = broken.list_tools().await.unwrap_err().to_string();
    assert!(
        err.contains("stderr: Error: Cannot find module tsx"),
        "{err}"
    );

    let missing = upstream(json!({"command": "/nonexistent/mcp-server"}), &store);
    let err = missing.list_tools().await.unwrap_err().to_string();
    assert!(
        err.contains("cannot start /nonexistent/mcp-server"),
        "{err}"
    );

    let no_dir = upstream(
        json!({"command": "/bin/sh", "cwd": "/nonexistent/dir"}),
        &store,
    );
    let err = no_dir.list_tools().await.unwrap_err().to_string();
    assert!(
        err.contains("cwd /nonexistent/dir is not a directory"),
        "{err}"
    );
}

#[tokio::test]
async fn leaves_a_failing_server_alone_for_a_while() {
    let log = StartLog::new("failing");
    let broken = upstream_with(
        json!({
            "command": "/bin/sh",
            "args": ["-c", "echo start >> \"$START_LOG\"; exit 1"],
            "env": {"START_LOG": log.0},
        }),
        &SecretStore::memory(),
        Timings {
            retry_delay: Duration::from_millis(400),
            ..Timings::default()
        },
    );
    assert!(broken.list_tools().await.is_err());
    assert_eq!(log.starts(), 1);

    // Asked again at once, by a list or a call: the same error, and no new process.
    let err = broken.list_tools().await.unwrap_err().to_string();
    assert!(err.contains("next attempt in 1s"), "{err}");
    assert!(broken.tools().await.is_none());
    assert!(call_text(&broken, "env").await.is_err());
    assert_eq!(log.starts(), 1);

    // After the delay it is tried again, and then left alone for twice as long.
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert!(broken.list_tools().await.is_err());
    assert_eq!(log.starts(), 2);
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert!(broken.list_tools().await.is_err());
    assert_eq!(log.starts(), 2);
}

#[tokio::test]
async fn stops_an_idle_server() {
    let log = StartLog::new("idle");
    let local = upstream_with(
        json!({
            "command": "/bin/sh",
            "args": [fixtures().join("stdio-server.sh")],
            "env": {"START_LOG": log.0},
            "idleTimeout": 1,
        }),
        &SecretStore::memory(),
        Timings {
            list_ttl: Duration::from_millis(100),
            ..Timings::default()
        },
    );

    // Nothing cached yet: the list starts it.
    assert_eq!(local.tools().await.unwrap().len(), 2);
    assert_eq!(log.starts(), 1);
    assert!(log.running());

    // A call longer than the idle time is not cut off.
    assert_eq!(call_text(&local, "slow").await.unwrap(), "slow done");
    assert_eq!(log.starts(), 1);
    assert!(log.running());

    // Left alone, it is stopped. Its tools are still listed, old as the list is, without
    // starting it again.
    tokio::time::sleep(Duration::from_millis(1500)).await;
    assert!(!log.running());
    assert_eq!(local.tools().await.unwrap().len(), 2);
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(log.starts(), 1);

    // The next call starts it again.
    assert!(call_text(&local, "env").await.is_ok());
    assert_eq!(log.starts(), 2);
    assert!(log.running());
    local.close().await;
    assert!(!log.running());
}

/// Wait until the cached tool list has `n` tools.
async fn cached_tools(upstream: &Upstream, n: usize) -> bool {
    for _ in 0..40 {
        if upstream.tools().await.is_some_and(|tools| tools.len() == n) {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    false
}

#[tokio::test]
async fn follows_a_changing_tool_list() {
    // Not a start log here, only a file that exists or not.
    let flag = StartLog::new("more");
    let local = upstream(
        json!({
            "command": "/bin/sh",
            "args": [fixtures().join("stdio-server.sh")],
            "env": {"MORE_FLAG": flag.0},
        }),
        &SecretStore::memory(),
    );
    assert_eq!(local.tools().await.unwrap().len(), 2);

    // The cached list is recent, so a change goes unnoticed...
    std::fs::write(&flag.0, "").unwrap();
    assert_eq!(local.tools().await.unwrap().len(), 2);
    // ...until the server says its tools changed.
    assert_eq!(call_text(&local, "notify").await.unwrap(), "notified");
    assert!(cached_tools(&local, 3).await);

    // A server that starts again may be another version: its tools are fetched again.
    std::fs::remove_file(&flag.0).unwrap();
    assert!(call_text(&local, "crash").await.is_err());
    assert!(call_text(&local, "env").await.is_ok());
    assert!(cached_tools(&local, 2).await);
    local.close().await;
}

#[tokio::test]
async fn replaces_a_server_that_stopped_answering() {
    let log = StartLog::new("hung");
    let local = upstream_with(
        json!({
            "command": "/bin/sh",
            "args": [fixtures().join("stdio-server.sh")],
            "env": {"START_LOG": log.0},
        }),
        &SecretStore::memory(),
        Timings {
            call_timeout: Duration::from_millis(300),
            ping_timeout: Duration::from_millis(700),
            ..Timings::default()
        },
    );

    // A call that takes too long fails, but the server still answers: it stays.
    let err = call_text(&local, "slow").await.unwrap_err().to_string();
    assert!(err.contains("timed out"), "{err}");
    tokio::time::sleep(Duration::from_millis(1000)).await;
    assert!(log.running());
    assert!(call_text(&local, "env").await.is_ok());
    assert_eq!(log.starts(), 1);

    // One that answers nothing any more is given up: the next call gets a new process...
    let err = call_text(&local, "hang").await.unwrap_err().to_string();
    assert!(err.contains("timed out"), "{err}");
    tokio::time::sleep(Duration::from_millis(1200)).await;
    assert!(call_text(&local, "env").await.is_ok());
    assert_eq!(log.starts(), 2);
    // ...and the old one is killed, after the few seconds it gets to exit by itself.
    for _ in 0..100 {
        if !log.nth_running(0) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(!log.nth_running(0));
    assert!(log.running());
    local.close().await;
}

#[tokio::test]
async fn stopping_a_server_stops_what_it_started() {
    let children = StartLog::new("children");
    let local = upstream(
        json!({
            "command": "/bin/sh",
            "args": [fixtures().join("stdio-server.sh")],
            "env": {"CHILD_LOG": children.0},
        }),
        &SecretStore::memory(),
    );
    assert_eq!(call_text(&local, "hog").await.unwrap(), "hogging");
    assert!(children.running());

    // The server itself exits as soon as it is asked to, leaving its child behind.
    local.close().await;
    assert!(!children.running());
}

#[tokio::test]
async fn stops_a_server_that_uses_too_much_memory() {
    let log = StartLog::new("large");
    let children = StartLog::new("large-children");
    let local = upstream_with(
        json!({
            "command": "/bin/sh",
            "args": [fixtures().join("stdio-server.sh")],
            "env": {"START_LOG": log.0, "CHILD_LOG": children.0},
            "maxMemoryMb": 20,
        }),
        &SecretStore::memory(),
        Timings {
            memory_check: Duration::from_millis(200),
            ..Timings::default()
        },
    );

    // Measured a few times and found small: it stays.
    assert!(call_text(&local, "env").await.is_ok());
    tokio::time::sleep(Duration::from_millis(900)).await;
    assert!(log.running());

    // What it starts counts too: with a child holding 30 MB it is stopped, child included.
    assert_eq!(call_text(&local, "hog").await.unwrap(), "hogging");
    for _ in 0..100 {
        if !log.running() && !children.running() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(!log.running());
    assert!(!children.running());

    // The next call starts it again.
    assert!(call_text(&local, "env").await.is_ok());
    assert_eq!(log.starts(), 2);
    local.close().await;
}

async fn call_for(upstream: &Upstream, tool: &str, session: &str) -> Result<String, UpstreamError> {
    let response = upstream
        .call_tool_for(
            CallToolRequestParams::new(tool.to_owned()),
            Caller {
                session: Some(session),
                ..Caller::default()
            },
            CancellationToken::new(),
        )
        .await?;
    let CallToolResponse::Complete(result) = response else {
        panic!("expected a complete result")
    };
    Ok(result.content[0].as_text().unwrap().text.clone())
}

/// Wait up to two seconds for `done`.
async fn eventually(mut done: impl FnMut() -> bool) -> bool {
    for _ in 0..40 {
        if done() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    done()
}

fn per_agent(log: &StartLog, more: serde_json::Value) -> Arc<Upstream> {
    let mut server = json!({
        "command": "/bin/sh",
        "args": [fixtures().join("stdio-server.sh")],
        "env": {"START_LOG": log.0},
        "perAgent": true,
    });
    server
        .as_object_mut()
        .unwrap()
        .extend(more.as_object().unwrap().clone());
    upstream_with(
        server,
        &SecretStore::memory(),
        Timings {
            agent_check: Duration::from_millis(100),
            ..Timings::default()
        },
    )
}

#[tokio::test]
async fn gives_each_agent_its_own_copy() {
    let log = StartLog::new("per-agent");
    let local = per_agent(&log, json!({}));

    // Listing starts the server once, to learn its tools, and stops it again.
    assert_eq!(local.tools().await.unwrap().len(), 2);
    assert_eq!(log.starts(), 1);
    assert!(
        eventually(|| !log.running()).await,
        "still running after listing"
    );

    // Each session gets a process of its own, which it keeps.
    assert!(call_for(&local, "env", "1.0.a").await.is_ok());
    assert!(call_for(&local, "env", "1.0.b").await.is_ok());
    assert!(call_for(&local, "env", "1.0.a").await.is_ok());
    assert_eq!(log.starts(), 3);
    assert!(log.nth_running(1) && log.nth_running(2));
    assert_eq!(local.copies(), 2);

    // Without a session there is nothing to call.
    let err = call_text(&local, "env").await.unwrap_err();
    assert!(err.to_string().contains("mcp-hub connect"), "{err}");

    // A session that ends stops its copy, and only its own.
    local.end_session("1.0.a");
    assert!(
        eventually(|| !log.nth_running(1)).await,
        "a's copy still runs"
    );
    assert!(log.nth_running(2));
    assert_eq!(local.copies(), 1);

    // A copy that crashes is started again for the same session.
    assert!(call_for(&local, "crash", "1.0.b").await.is_err());
    assert!(call_for(&local, "env", "1.0.b").await.is_ok());
    assert_eq!(log.starts(), 4);

    local.close().await;
    assert!(!log.nth_running(3));
    assert_eq!(local.copies(), 0);
}

#[tokio::test]
async fn stops_the_copy_of_an_agent_that_is_gone() {
    let log = StartLog::new("agent-gone");
    let local = per_agent(&log, json!({}));
    // An agent process that will go away, in this process's pid namespace.
    let mut agent = std::process::Command::new("sleep")
        .arg("30")
        .spawn()
        .unwrap();
    let session = format!("{}.{}.x", agent.id(), mcp_hub::connect::pid_namespace());
    assert!(call_for(&local, "env", &session).await.is_ok());
    // One in another pid namespace cannot be checked: it is left to its idle time.
    assert!(call_for(&local, "env", "1.999.y").await.is_ok());
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(log.running());
    assert_eq!(local.copies(), 2);

    agent.kill().unwrap();
    agent.wait().unwrap();
    assert!(
        eventually(|| local.copies() == 1).await,
        "copy of a gone agent kept"
    );
    assert!(
        eventually(|| !log.nth_running(0)).await,
        "its process still runs"
    );
    assert!(log.nth_running(1));
    local.close().await;
}

#[tokio::test]
async fn copies_stop_when_idle_and_are_capped() {
    let log = StartLog::new("copies-idle");
    let local = per_agent(&log, json!({"idleTimeout": 1, "maxCopies": 2}));
    assert!(call_for(&local, "env", "1.0.a").await.is_ok());
    assert!(call_for(&local, "env", "1.0.b").await.is_ok());
    let err = call_for(&local, "env", "1.0.c").await.unwrap_err();
    assert!(err.to_string().contains("maxCopies"), "{err}");
    assert_eq!(log.starts(), 2);

    // Idle, both stop and are forgotten; a new session then has room.
    tokio::time::sleep(Duration::from_millis(1300)).await;
    assert!(!log.nth_running(0) && !log.nth_running(1));
    assert!(eventually(|| local.copies() == 0).await);
    assert!(call_for(&local, "env", "1.0.c").await.is_ok());
    assert_eq!(log.starts(), 3);
    local.close().await;
}

#[tokio::test]
async fn a_dropped_server_leaves_no_process_behind() {
    let log = StartLog::new("dropped");
    let children = StartLog::new("dropped-children");
    let local = per_agent(
        &log,
        json!({"env": {"START_LOG": log.0, "CHILD_LOG": children.0}}),
    );
    assert_eq!(call_for(&local, "hog", "1.0.a").await.unwrap(), "hogging");
    assert!(log.running() && children.running());
    // Never closed: dropping it is enough.
    drop(local);
    assert!(eventually(|| !log.running() && !children.running()).await);
}
