# Setting Up CC Slack Bot

## Step 1: Create Slack App

1. Go to https://api.slack.com/apps
2. Click **Create New App**
3. Choose **From scratch**
4. App Name: `cc`
5. Pick your workspace (maccosmos)

## Step 2: Enable Socket Mode

1. Go to **Socket Mode** in the left sidebar
2. Enable Socket Mode
3. Click **Generate Token** to create an App-Level Token
4. Copy the App-Level Token (starts with `xapp-`)

## Step 3: Configure Bot Permissions

1. Go to **OAuth & Permissions** in the left sidebar
2. Add these **Bot Token Scopes**:
   - `chat:write` - Send messages
   - `channels:history` - Read channel messages
   - `channels:read` - List channels
   - `groups:history` - Read group messages
   - `groups:read` - List groups
   - `im:history` - Read direct messages
   - `im:read` - List DMs
   - `users:read` - Get user info

3. Click **Install to Workspace** at the top
4. Copy the Bot Token (starts with `xoxb-`)

## Step 4: Subscribe to Events

1. Go to **Event Subscriptions** in the left sidebar
2. Subscribe to these **Bot Events**:
   - `message.channels`
   - `message.groups`
   - `message.im`

## Step 5: Add Bot to Channel

1. Go to your Slack channel
2. Right-click channel name → **View channel details**
3. Go to **Integrations** → **Add apps**
4. Add the `cc` bot

## Step 6: Provide Tokens

Please provide the following tokens:

| Token Type | Example |
|------------|---------|
| Bot Token | `xoxb-...` |
| App Token | `xapp-...` |

---

Once you provide these tokens, I'll update the nanoclaw configuration.
