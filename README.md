# Brickbase RWA Investor Agent

An AI agent for investing in tokenised real estate (RWA). Connects to platforms via MCP, discovers assets, evaluates them against your criteria, and executes purchases.

## Overview

The agent is platform-agnostic: it discovers tools and resources from any compatible MCP server at runtime. No prior knowledge of the platform is required.

**Flow:**

1. **Criteria discovery** — Multi-turn conversation to capture yield target, max price, locations, risk tolerance, and platform URL
2. **Platform connection** — Connects to the MCP server at the provided URL
3. **Capability mapping** — Uses the LLM to map discovered tools to fetch assets, fetch detail, and purchase
4. **Evaluation** — Fetches assets, evaluates each against your criteria (prices normalised from on-chain e6)
5. **Report** — Writes a Markdown report of planned decisions before execution
6. **Execution** — Signs and broadcasts purchase transactions via the discovered tools

## Requirements

- Node.js 18+
- A platform MCP server (e.g. Brickbase)
- Ethereum wallet with USDC and gas
- Gemini or DeepSeek API key

## Setup

```bash
npm install
cp .env.example .env
# Edit .env with your keys and config
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | Yes* | Gemini API key |
| `DEEPSEEK_API_KEY` | Yes* | DeepSeek API key (if using DeepSeek) |
| `GEMINI_MODEL` | No | Gemini model (default: `gemini-2.0-flash`) |
| `DEEPSEEK_MODEL` | No | DeepSeek model (default: `deepseek-chat`) |
| `AGENT_PRIVATE_KEY` | Yes | Wallet private key (0x-prefixed) |
| `MCP_URL` | Yes | Platform MCP URL (e.g. `http://localhost:3000/mcp`) |
| `RPC_URL` | No | Ethereum RPC (default: `http://localhost:8545`) |
| `REPORT_DIR` | No | Output directory for reports (default: `./reports`) |

\* Use either Gemini or DeepSeek depending on agent configuration.

## Usage

```bash
npx tsx src/agent.ts
```

The agent will prompt you for investment criteria, then connect to the platform, evaluate assets, and execute purchases.

## Reports

Reports are written to `./reports` (or `REPORT_DIR`) as Markdown files. Each run produces a report with:

- Investment criteria
- Per-asset decisions (PURCHASE/REJECT) and rationale
- Transaction hashes for executed purchases

## Price Handling

On-chain prices use USDC decimals (e6). The agent normalises these to human-readable values before evaluation and reporting.

## License

ISC
