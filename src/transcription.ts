import { execFile } from 'child_process';
import { promisify } from 'util';
import { writeFile, unlink, readFile } from 'fs/promises';
import * as path from 'path';
import { tmpdir } from 'os';
import { randomUUID } from 'crypto';

import { logger } from './logger.js';

const execFileAsync = promisify(execFile);

/**
 * Local whisper transcription using the OpenAI whisper Python package.
 * Falls back to whisper.cpp (whisper-cli) if available.
 *
 * No API key required - runs entirely on-device.
 */

// Check which whisper binary is available
const WHISPER_BIN = process.env.WHISPER_BIN || '/opt/homebrew/bin/whisper';
const WHISPER_CPP_BIN = process.env.WHISPER_CPP_BIN || 'whisper-cli';

// Model for local whisper (will be downloaded on first use if using Python whisper)
const WHISPER_MODEL = process.env.WHISPER_MODEL || 'base';

// whisper.cpp model path (required for whisper.cpp)
const WHISPER_CPP_MODEL =
  process.env.WHISPER_CPP_MODEL || 'data/models/ggml-base.bin';

interface TranscriptionResult {
  text: string;
  success: boolean;
  error?: string;
}

/**
 * Transcribe an audio file using local whisper.
 * Tries Python whisper first, falls back to whisper.cpp.
 */
export async function transcribeAudio(
  audioBuffer: Buffer,
  mimeType: string,
): Promise<TranscriptionResult> {
  // Generate temp file paths
  const id = randomUUID();
  const inputExt = getExtensionFromMime(mimeType);
  const inputPath = path.join(tmpdir(), `nanoclaw-input-${id}.${inputExt}`);
  const outputDir = tmpdir();
  const outputPath = path.join(outputDir, `nanoclaw-input-${id}.txt`);

  try {
    // Write audio buffer to temp file
    await writeFile(inputPath, audioBuffer);
    logger.debug(
      { path: inputPath, size: audioBuffer.length, mimeType },
      'Wrote audio file for transcription',
    );

    // Try Python whisper first (it's installed at /opt/homebrew/bin/whisper)
    try {
      const result = await transcribeWithPythonWhisper(
        inputPath,
        outputDir,
        outputPath,
      );
      if (result.success) {
        return result;
      }
    } catch (err) {
      logger.debug({ err }, 'Python whisper failed, trying whisper.cpp');
    }

    // Fall back to whisper.cpp
    const result = await transcribeWithWhisperCpp(
      inputPath,
      outputDir,
      outputPath,
    );
    return result;
  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : String(err);
    logger.error({ err }, 'Local transcription failed');
    return { text: '', success: false, error: errorMessage };
  } finally {
    // Clean up temp files
    await cleanup([inputPath, outputPath]);
  }
}

/**
 * Transcribe using Python whisper package.
 * Downloads model on first use.
 */
async function transcribeWithPythonWhisper(
  inputPath: string,
  outputDir: string,
  outputPath: string,
): Promise<TranscriptionResult> {
  try {
    // Python whisper outputs to inputPath.txt by default
    const { stdout, stderr } = await execFileAsync(
      WHISPER_BIN,
      [
        inputPath,
        '--model',
        WHISPER_MODEL,
        '--output_dir',
        outputDir,
        '--output_format',
        'txt',
        '--verbose',
        'False',
      ],
      { timeout: 120000 }, // 2 minute timeout
    );

    // Read the output file (whisper appends .txt to the input filename)
    const actualOutputPath = `${inputPath}.txt`;
    const text = (await readFile(actualOutputPath, 'utf-8')).trim();

    // Clean up the generated txt file
    await unlink(actualOutputPath).catch(() => {});

    if (text) {
      logger.info(
        { length: text.length },
        'Transcribed audio with Python whisper',
      );
      return { text, success: true };
    }

    return { text: '', success: false, error: 'Empty transcription result' };
  } catch (err) {
    throw err;
  }
}

/**
 * Transcribe using whisper.cpp (whisper-cli).
 * Requires pre-downloaded GGML model.
 */
async function transcribeWithWhisperCpp(
  inputPath: string,
  outputDir: string,
  outputPath: string,
): Promise<TranscriptionResult> {
  try {
    const { stdout } = await execFileAsync(
      WHISPER_CPP_BIN,
      [
        '-m',
        WHISPER_CPP_MODEL,
        '-f',
        inputPath,
        '--output-txt',
        '--no-timestamps',
      ],
      { timeout: 120000 },
    );

    // whisper-cli outputs to inputPath.txt
    const actualOutputPath = `${inputPath}.txt`;
    const text = (await readFile(actualOutputPath, 'utf-8')).trim();

    // Clean up
    await unlink(actualOutputPath).catch(() => {});

    if (text) {
      logger.info(
        { length: text.length },
        'Transcribed audio with whisper.cpp',
      );
      return { text, success: true };
    }

    return { text: '', success: false, error: 'Empty whisper.cpp result' };
  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : String(err);
    return { text: '', success: false, error: errorMessage };
  }
}

/**
 * Get file extension from MIME type.
 */
function getExtensionFromMime(mimeType: string): string {
  const mimeMap: Record<string, string> = {
    'audio/mp4': 'mp4',
    'audio/m4a': 'm4a',
    'audio/mpeg': 'mp3',
    'audio/wav': 'wav',
    'audio/webm': 'webm',
    'audio/ogg': 'ogg',
    'audio/aac': 'aac',
    'video/mp4': 'mp4', // Slack voice messages come as video/mp4
  };
  return mimeMap[mimeType] || 'mp4';
}

/**
 * Clean up temp files.
 */
async function cleanup(paths: string[]): Promise<void> {
  for (const p of paths) {
    try {
      await unlink(p);
    } catch {
      // Ignore cleanup errors
    }
  }
}
