/**
 * Host Runner for NanoClaw
 * Runs Claude directly on the host without container isolation
 * WARNING: This gives Claude full system access - use with caution!
 */

import { ChildProcess, spawn } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';

import {
  CONTAINER_MAX_OUTPUT_SIZE,
  CONTAINER_TIMEOUT,
  DATA_DIR,
  GROUPS_DIR,
  IDLE_TIMEOUT,
  TIMEZONE,
} from './config.js';
import { resolveGroupFolderPath, resolveGroupIpcPath } from './group-folder.js';
import { logger } from './logger.js';
import { RegisteredGroup } from './types.js';

const OUTPUT_START_MARKER = '---NANOCLAW_OUTPUT_START---';
const OUTPUT_END_MARKER = '---NANOCLAW_OUTPUT_END---';

// Slack conversation memory system
interface SlackMemory {
  lastUpdated: string;
  conversationSummary: string;
  recentTopics: string[];
  userPreferences: Record<string, string>;
  pendingTasks: string[];
}

const MEMORY_MAX_AGE = 7 * 24 * 60 * 60 * 1000; // 7 days

function getSlackMemoryPath(groupFolder: string): string {
  return path.join(
    DATA_DIR,
    'sessions',
    groupFolder,
    '.claude',
    'slack_memory.json',
  );
}

function loadSlackMemory(groupFolder: string): SlackMemory {
  const memoryPath = getSlackMemoryPath(groupFolder);
  try {
    if (fs.existsSync(memoryPath)) {
      const data = JSON.parse(fs.readFileSync(memoryPath, 'utf-8'));
      // Check if memory is too old
      const lastUpdated = new Date(data.lastUpdated).getTime();
      if (Date.now() - lastUpdated > MEMORY_MAX_AGE) {
        logger.debug({ groupFolder }, 'Slack memory expired, resetting');
        return createEmptyMemory();
      }
      return data;
    }
  } catch (err) {
    logger.warn(
      { groupFolder, err },
      'Failed to load Slack memory, creating new',
    );
  }
  return createEmptyMemory();
}

function createEmptyMemory(): SlackMemory {
  return {
    lastUpdated: new Date().toISOString(),
    conversationSummary: '',
    recentTopics: [],
    userPreferences: {},
    pendingTasks: [],
  };
}

function saveSlackMemory(groupFolder: string, memory: SlackMemory): void {
  const memoryPath = getSlackMemoryPath(groupFolder);
  memory.lastUpdated = new Date().toISOString();
  try {
    fs.mkdirSync(path.dirname(memoryPath), { recursive: true });
    fs.writeFileSync(memoryPath, JSON.stringify(memory, null, 2));
  } catch (err) {
    logger.warn({ groupFolder, err }, 'Failed to save Slack memory');
  }
}

function buildMemoryContext(memory: SlackMemory): string {
  const parts: string[] = [];

  if (memory.conversationSummary) {
    parts.push(`## Conversation Summary\n${memory.conversationSummary}`);
  }

  if (memory.recentTopics.length > 0) {
    parts.push(
      `## Recent Topics\n${memory.recentTopics.map((t) => `- ${t}`).join('\n')}`,
    );
  }

  if (Object.keys(memory.userPreferences).length > 0) {
    parts.push(
      `## User Preferences\n${Object.entries(memory.userPreferences)
        .map(([k, v]) => `- ${k}: ${v}`)
        .join('\n')}`,
    );
  }

  if (memory.pendingTasks.length > 0) {
    parts.push(
      `## Pending Tasks\n${memory.pendingTasks.map((t) => `- ${t}`).join('\n')}`,
    );
  }

  return parts.length > 0
    ? `<slack_memory>\n${parts.join('\n\n')}\n</slack_memory>`
    : '';
}

function updateMemoryFromConversation(
  memory: SlackMemory,
  prompt: string,
  response: string,
): void {
  // Extract topics from the conversation
  const topics: string[] = [];

  // Simple topic extraction - look for key phrases
  const topicPatterns = [
    /(?:working on|building|creating|implementing|fixing)\s+([^.!?\n]{5,50})/gi,
    /(?:task|project|feature)\s*:\s*([^.!?\n]{5,50})/gi,
  ];

  for (const pattern of topicPatterns) {
    const matches = [
      ...prompt.matchAll(pattern),
      ...response.matchAll(pattern),
    ];
    for (const match of matches) {
      if (match[1] && !topics.includes(match[1])) {
        topics.push(match[1].trim());
      }
    }
  }

  // Update recent topics (keep last 10)
  memory.recentTopics = [...new Set([...topics, ...memory.recentTopics])].slice(
    0,
    10,
  );

  // Update conversation summary (keep last 500 chars of context)
  const newSummary = `Last exchange: User asked about topics in the conversation. Assistant provided help.`;
  if (memory.conversationSummary.length > 2000) {
    memory.conversationSummary =
      memory.conversationSummary.slice(-1500) + '\n' + newSummary;
  } else {
    memory.conversationSummary = memory.conversationSummary
      ? memory.conversationSummary + '\n' + newSummary
      : newSummary;
  }

  // Extract pending tasks
  const taskPattern =
    /(?:todo|task|pending|need to|should|will)\s*:\s*([^.!?\n]{5,100})/gi;
  const taskMatches = response.matchAll(taskPattern);
  for (const match of taskMatches) {
    if (match[1] && !memory.pendingTasks.includes(match[1])) {
      memory.pendingTasks.push(match[1].trim());
    }
  }
  // Keep only last 5 pending tasks
  memory.pendingTasks = memory.pendingTasks.slice(-5);
}

export interface HostInput {
  prompt: string;
  sessionId?: string;
  groupFolder: string;
  chatJid: string;
  isMain: boolean;
  isScheduledTask?: boolean;
  assistantName?: string;
}

export interface HostOutput {
  status: 'success' | 'error';
  result: string | null;
  newSessionId?: string;
  error?: string;
}

function setupGroupEnvironment(
  group: RegisteredGroup,
  input: HostInput,
): {
  cwd: string;
  env: NodeJS.ProcessEnv;
  claudeDir: string;
} {
  const groupDir = resolveGroupFolderPath(group.folder);
  fs.mkdirSync(groupDir, { recursive: true });

  // Per-group Claude sessions directory
  const claudeDir = path.join(DATA_DIR, 'sessions', group.folder, '.claude');
  fs.mkdirSync(claudeDir, { recursive: true });

  // Create settings.json if not exists
  const settingsFile = path.join(claudeDir, 'settings.json');
  if (!fs.existsSync(settingsFile)) {
    fs.writeFileSync(
      settingsFile,
      JSON.stringify(
        {
          env: {
            CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS: '1',
            CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD: '1',
            CLAUDE_CODE_DISABLE_AUTO_MEMORY: '0',
          },
        },
        null,
        2,
      ) + '\n',
    );
  }

  // Copy skills from container/skills/
  const skillsSrc = path.join(process.cwd(), 'container', 'skills');
  const skillsDst = path.join(claudeDir, 'skills');
  if (fs.existsSync(skillsSrc)) {
    for (const skillDir of fs.readdirSync(skillsSrc)) {
      const srcDir = path.join(skillsSrc, skillDir);
      if (!fs.statSync(srcDir).isDirectory()) continue;
      const dstDir = path.join(skillsDst, skillDir);
      fs.cpSync(srcDir, dstDir, { recursive: true });
    }
  }

  // Setup IPC directory
  const groupIpcDir = resolveGroupIpcPath(group.folder);
  fs.mkdirSync(path.join(groupIpcDir, 'messages'), { recursive: true });
  fs.mkdirSync(path.join(groupIpcDir, 'tasks'), { recursive: true });
  fs.mkdirSync(path.join(groupIpcDir, 'input'), { recursive: true });

  // Build environment
  // IMPORTANT: Unset CLAUDECODE to allow spawning nested Claude sessions
  // Otherwise Claude refuses to run inside another Claude Code session
  const { CLAUDECODE, ...restEnv } = process.env;
  const env: NodeJS.ProcessEnv = {
    ...restEnv,
    HOME: os.homedir(),
    TZ: TIMEZONE,
    // Claude settings
    CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS: '1',
    CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD: '1',
    CLAUDE_CODE_DISABLE_AUTO_MEMORY: '0',
    // Allow dangerous permissions for full system access
    CLAUDE_CODE_DISABLE_SANDBOX: '1',
    // Working directory context
    PWD: input.isMain ? process.cwd() : groupDir,
  };

  // For main group, also expose project root
  if (input.isMain) {
    env['NANOCLAW_PROJECT_ROOT'] = process.cwd();
    env['NANOCLAW_IS_MAIN'] = '1';
  }

  return {
    cwd: groupDir,
    env,
    claudeDir,
  };
}

export async function runHostAgent(
  group: RegisteredGroup,
  input: HostInput,
  onProcess: (proc: ChildProcess) => void,
  onOutput?: (output: HostOutput) => Promise<void>,
): Promise<HostOutput> {
  const startTime = Date.now();

  const { cwd, env, claudeDir } = setupGroupEnvironment(group, input);
  const groupDir = resolveGroupFolderPath(group.folder);
  const logsDir = path.join(groupDir, 'logs');
  fs.mkdirSync(logsDir, { recursive: true });

  // Load Slack conversation memory
  const slackMemory = loadSlackMemory(group.folder);
  const memoryContext = buildMemoryContext(slackMemory);

  logger.info(
    {
      group: group.name,
      isMain: input.isMain,
      cwd,
      hasMemory: !!memoryContext,
    },
    'Starting host agent (no container)',
  );

  // Build the prompt with session context and memory
  let fullPrompt = input.prompt;
  if (input.sessionId) {
    fullPrompt = `[Continuing session: ${input.sessionId}]\n\n${fullPrompt}`;
  }
  if (memoryContext) {
    fullPrompt = `${memoryContext}\n\n---\n\n${fullPrompt}`;
  }

  return new Promise((resolve) => {
    // Spawn claude directly on host
    // Use --print for non-interactive mode
    // Use --dangerously-skip-permissions to bypass all approval prompts
    const claude = spawn(
      'claude',
      ['--print', '--dangerously-skip-permissions'],
      {
        cwd,
        env: {
          ...env,
          HOME: claudeDir, // Use group's .claude directory
        },
        stdio: ['pipe', 'pipe', 'pipe'],
      },
    );

    onProcess(claude);

    // Write prompt to stdin
    claude.stdin.write(fullPrompt);
    claude.stdin.end();

    let stdout = '';
    let stderr = '';
    let stdoutTruncated = false;
    let parseBuffer = '';
    let newSessionId: string | undefined;
    let outputChain = Promise.resolve();
    let hadStreamingOutput = false;

    claude.stdout.on('data', (data) => {
      const chunk = data.toString();

      if (!stdoutTruncated) {
        const remaining = CONTAINER_MAX_OUTPUT_SIZE - stdout.length;
        if (chunk.length > remaining) {
          stdout += chunk.slice(0, remaining);
          stdoutTruncated = true;
        } else {
          stdout += chunk;
        }
      }

      // Stream-parse for output markers
      if (onOutput) {
        parseBuffer += chunk;
        let startIdx: number;
        while ((startIdx = parseBuffer.indexOf(OUTPUT_START_MARKER)) !== -1) {
          const endIdx = parseBuffer.indexOf(OUTPUT_END_MARKER, startIdx);
          if (endIdx === -1) break;

          const jsonStr = parseBuffer
            .slice(startIdx + OUTPUT_START_MARKER.length, endIdx)
            .trim();
          parseBuffer = parseBuffer.slice(endIdx + OUTPUT_END_MARKER.length);

          try {
            const parsed: HostOutput = JSON.parse(jsonStr);
            if (parsed.newSessionId) {
              newSessionId = parsed.newSessionId;
            }
            hadStreamingOutput = true;
            resetTimeout();
            outputChain = outputChain.then(() => onOutput(parsed));
          } catch (err) {
            logger.warn(
              { group: group.name, error: err },
              'Failed to parse output chunk',
            );
          }
        }
      }
    });

    claude.stderr.on('data', (data) => {
      const chunk = data.toString();
      for (const line of chunk.trim().split('\n')) {
        if (line) logger.debug({ host: group.folder }, line);
      }
      if (stderr.length < CONTAINER_MAX_OUTPUT_SIZE) {
        stderr += chunk;
      }
    });

    let timedOut = false;
    const configTimeout = group.containerConfig?.timeout || CONTAINER_TIMEOUT;
    const timeoutMs = Math.max(configTimeout, IDLE_TIMEOUT + 30_000);

    const killOnTimeout = () => {
      timedOut = true;
      logger.error({ group: group.name }, 'Host agent timeout, killing');
      claude.kill('SIGTERM');
    };

    let timeout = setTimeout(killOnTimeout, timeoutMs);

    const resetTimeout = () => {
      clearTimeout(timeout);
      timeout = setTimeout(killOnTimeout, timeoutMs);
    };

    claude.on('close', (code) => {
      clearTimeout(timeout);
      const duration = Date.now() - startTime;

      // Write log
      const ts = new Date().toISOString().replace(/[:.]/g, '-');
      const logFile = path.join(logsDir, `host-${ts}.log`);
      fs.writeFileSync(
        logFile,
        [
          `=== Host Run Log ===`,
          `Timestamp: ${new Date().toISOString()}`,
          `Group: ${group.name}`,
          `IsMain: ${input.isMain}`,
          `Duration: ${duration}ms`,
          `Exit Code: ${code}`,
          `CWD: ${cwd}`,
          `Timed Out: ${timedOut}`,
          ``,
          `=== Stderr ===`,
          stderr.slice(-2000),
          ``,
          `=== Stdout ===`,
          stdout.slice(-2000),
        ].join('\n'),
      );

      if (timedOut && hadStreamingOutput) {
        logger.info(
          { group: group.name, duration },
          'Host agent timed out after output',
        );
        outputChain.then(() =>
          resolve({ status: 'success', result: null, newSessionId }),
        );
        return;
      }

      if (timedOut) {
        resolve({
          status: 'error',
          result: null,
          error: `Timed out after ${configTimeout}ms`,
        });
        return;
      }

      if (code !== 0) {
        resolve({
          status: 'error',
          result: null,
          error: `Exit code ${code}: ${stderr.slice(-200)}`,
        });
        return;
      }

      if (onOutput) {
        // When using streaming mode, call onOutput with the result
        // If no streaming markers were found, treat stdout as the result
        const trimmedOutput = stdout.trim();
        logger.debug(
          {
            group: group.name,
            hadStreamingOutput,
            outputLength: trimmedOutput.length,
          },
          'Host agent finished, checking output',
        );
        if (trimmedOutput && !hadStreamingOutput) {
          // No markers found - treat entire output as a single result
          const output: HostOutput = {
            status: 'success',
            result: trimmedOutput,
          };
          logger.info(
            { group: group.name, output: trimmedOutput.slice(0, 100) },
            'Calling onOutput with result',
          );
          outputChain = outputChain.then(() => onOutput(output));
          // Update and save memory after successful output
          updateMemoryFromConversation(slackMemory, fullPrompt, trimmedOutput);
          saveSlackMemory(group.folder, slackMemory);
        }
        outputChain.then(() => {
          logger.info({ group: group.name, duration }, 'Host agent completed');
          resolve({ status: 'success', result: null, newSessionId });
        });
        return;
      }

      // Parse output
      try {
        const startIdx = stdout.indexOf(OUTPUT_START_MARKER);
        const endIdx = stdout.indexOf(OUTPUT_END_MARKER);
        if (startIdx !== -1 && endIdx !== -1 && endIdx > startIdx) {
          const jsonStr = stdout
            .slice(startIdx + OUTPUT_START_MARKER.length, endIdx)
            .trim();
          const output: HostOutput = JSON.parse(jsonStr);
          // Update and save memory
          if (output.result) {
            updateMemoryFromConversation(
              slackMemory,
              fullPrompt,
              output.result,
            );
            saveSlackMemory(group.folder, slackMemory);
          }
          logger.info(
            { group: group.name, duration, status: output.status },
            'Host agent completed',
          );
          resolve(output);
        } else {
          // No markers - treat entire output as result
          const result = stdout.trim() || null;
          if (result) {
            updateMemoryFromConversation(slackMemory, fullPrompt, result);
            saveSlackMemory(group.folder, slackMemory);
          }
          resolve({ status: 'success', result });
        }
      } catch (err) {
        resolve({ status: 'success', result: stdout.trim() || null });
      }
    });

    claude.on('error', (err) => {
      clearTimeout(timeout);
      logger.error({ group: group.name, error: err }, 'Host agent spawn error');
      resolve({
        status: 'error',
        result: null,
        error: `Spawn error: ${err.message}`,
      });
    });
  });
}
