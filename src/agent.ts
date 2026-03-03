/**
 * ============================================================
 * Brickbase Investor Agent
 * ============================================================
 *
 * A single-run agent acting on behalf of an investor that:
 *
 *  1. CRITERIA DISCOVERY
 *     Multi-turn Gemini conversation to elicit the investor's
 *     purchase conditions: yield target, max price-per-share (USDC),
 *     location preferences, risk appetite, etc.
 *
 *  2. PROPERTY REVIEW
 *     Fetches every property in the AssetVault via the brickbase
 *     MCP server (get_property_list + get_property_detail).
 *     Contract addresses come from the MCP resource
 *     config://deployments — no ASSET_SHARES_ADDRESS env var needed.
 *
 *  3. EVALUATION
 *     Asks Gemini to assess each property against the agreed criteria
 *     and decide: PURCHASE or REJECT with explicit data-backed reasons.
 *
 *  4. EXECUTION
 *     For each PURCHASE decision, calls purchase_shares on the MCP
 *     server to obtain unsigned tx payloads (USDC approve +
 *     purchaseShares), then signs and broadcasts them with the
 *     investor's wallet via ethers. The MCP server holds no private keys.
 *
 *  5. REPORT
 *     Writes a Markdown report summarising every property reviewed:
 *     decision, shares purchased, USDC cost, and Gemini rationale.
 *
 * ── Environment variables ──────────────────────────────────────
 *   GOOGLE_API_KEY      Gemini API key
 *   AGENT_PRIVATE_KEY   Investor wallet private key (0x-prefixed)
 *   MCP_URL             MCP server URL (optional). If set, use HTTP. If not set, spawn stdio server.
 *   INVESTOR_NAME       Report display name (default: Investor)
 *   REPORT_DIR          Output directory   (default: ./reports)
 *
 * ── Install ────────────────────────────────────────────────────
 *   npm install @google/genai @modelcontextprotocol/sdk ethers dotenv
 *   npm install -D typescript tsx @types/node
 *
 * ── Run ────────────────────────────────────────────────────────
 *   npx tsx investorAgent.ts
 * ============================================================
 */

import dotenv from "dotenv";
import { GoogleGenAI, type Content } from "@google/genai";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { ethers } from "ethers";
import * as fs from "fs/promises";
import * as path from "path";
import * as readline from "readline/promises";

dotenv.config();

// ─── Types ────────────────────────────────────────────────────────────────────

interface InvestmentCriteria {
  summary: string;
  minYieldPercent?: number;
  maxPricePerShareUsdc?: number;
  preferredLocations?: string[];
  maxSharesPerProperty?: number;
  defaultShareCount: number;
  riskTolerance: "low" | "medium" | "high";
  additionalConditions: string[];
}

interface PropertyDetail {
  id: number | string;
  name: string;
  location?: string;
  pricePerShare?: number | string;
  totalShares?: number | string;
  availableShares?: number | string;
  yieldPercent?: number | string;
  occupancyRate?: number | string;
  [key: string]: unknown;
}

interface UnsignedTx {
  to: string;
  data: string;
  value?: string;
}

interface PurchasePayload {
  transactions: UnsignedTx[];
}

interface EvaluationResult {
  decision: "PURCHASE" | "REJECT";
  shareCount: number;
  reasoning: string;
  metricsAssessed: Record<string, string>;
}

interface PropertyDecision {
  property: PropertyDetail;
  decision: "PURCHASE" | "REJECT";
  shareCount: number;
  pricePerShare: string;
  totalCostUsdc: string;
  reasoning: string;
  metricsAssessed: Record<string, string>;
  txHashes: string[];
  error?: string;
}

interface AgentReport {
  investorName: string;
  walletAddress: string;
  timestamp: string;
  criteria: InvestmentCriteria;
  oraclePrices: Record<string, unknown>;
  deployments: Record<string, unknown>;
  decisions: PropertyDecision[];
  planned?: boolean;
  summary: {
    totalProperties: number;
    purchased: number;
    rejected: number;
    totalSharesAcquired: number;
    totalUsdcSpent: string;
  };
}

// ─── Config ───────────────────────────────────────────────────────────────────

const CONFIG = {
  gemini: {
    model: "gemini-3-flash-preview",
    maxOutputTokens: 2048,
    temperature: 0.4,
  },
  mcp: {
    url: process.env.MCP_URL ?? "",
    brickbaseRoot: path.resolve(process.cwd(), "../brickbase"),
  },
  wallet: {
    privateKey: process.env.AGENT_PRIVATE_KEY ?? "",
    rpcUrl: process.env.RPC_URL ?? "http://localhost:8545",
  },
  investorName: process.env.INVESTOR_NAME ?? "Investor",
  reportDir: process.env.REPORT_DIR ?? "./reports",
};

// ─── Logging ──────────────────────────────────────────────────────────────────

type LogLevel = "INFO" | "WARN" | "ERROR" | "STEP";

function log(level: LogLevel, msg: string): void {
  const icons: Record<LogLevel, string> = {
    INFO: "i",
    WARN: "!",
    ERROR: "x",
    STEP: ">",
  };
  console.log(`[${new Date().toISOString()}] [${icons[level]}]  ${msg}`);
}

const sleep = (ms: number): Promise<void> =>
  new Promise((r) => setTimeout(r, ms));

// ─── Gemini ───────────────────────────────────────────────────────────────────

/**
 * Initialise @google/genai.
 *
 * GoogleGenAI is the single, centralised entry-point in the new SDK —
 * replacing the fragmented per-model classes in @google/generative-ai.
 * All generate calls go through genAI.models.generateContent.
 */
function initGemini(): GoogleGenAI {
  const apiKey = process.env.GOOGLE_API_KEY;
  if (!apiKey) throw new Error("GOOGLE_API_KEY is not set.");
  log("INFO", "Gemini AI client ready");
  return new GoogleGenAI({ apiKey });
}

/** Send a multi-turn conversation to Gemini and return the text reply. */
async function geminiChat(
  genAI: GoogleGenAI,
  history: Content[],
  systemInstruction: string
): Promise<string> {
  const response = await genAI.models.generateContent({
    model: CONFIG.gemini.model,
    contents: history,
    config: {
      maxOutputTokens: CONFIG.gemini.maxOutputTokens,
      temperature: CONFIG.gemini.temperature,
      systemInstruction,
    },
  });
  return response.text ?? "";
}

/** Ask Gemini for a JSON response, strip any fences, and parse it. */
async function geminiJSON<T>(
  genAI: GoogleGenAI,
  prompt: string,
  systemInstruction: string
): Promise<T> {
  const raw = await geminiChat(
    genAI,
    [{ role: "user", parts: [{ text: prompt }] }],
    systemInstruction
  );
  const cleaned = raw
    .replace(/```json\s*/gi, "")
    .replace(/```\s*/g, "")
    .trim();
  try {
    return JSON.parse(cleaned) as T;
  } catch {
    throw new Error(`Gemini returned non-JSON:\n${raw}`);
  }
}

// ─── MCP ──────────────────────────────────────────────────────────────────────

/** Connect to the MCP server. Uses MCP_URL if set (HTTP); otherwise spawns stdio server. */
async function initMCP(): Promise<Client> {
  const client = new Client({
    name: "brickbase-investor-agent",
    version: "1.0.0",
  });
  if (CONFIG.mcp.url) {
    try {
      await client.connect(
        new StreamableHTTPClientTransport(new URL(CONFIG.mcp.url)) as Parameters<
          Client["connect"]
        >[0]
      );
    } catch (err) {
      const url = CONFIG.mcp.url;
      const code = (err as { code?: number })?.code;
      const msg =
        code === 404
          ? `MCP endpoint not found at ${url}.`
          : code && code >= 500
            ? `MCP server at ${url} returned error ${code}. Unset MCP_URL to use the stdio MCP server instead.`
            : `MCP server is not reachable at ${url}.`;
      throw new Error(msg, { cause: err });
    }
  } else {
    await client.connect(
      new StdioClientTransport({
        command: "npx",
        args: ["nx", "run", "mcp:serve"],
        cwd: CONFIG.mcp.brickbaseRoot,
        env: process.env as Record<string, string>,
      })
    );
  }
  log("INFO", "MCP server connected");
  return client;
}

/** Call an MCP tool and return the parsed JSON result. */
async function mcpTool<T = unknown>(
  client: Client,
  toolName: string,
  args: Record<string, unknown> = {}
): Promise<T> {
  const result = await client.callTool({ name: toolName, arguments: args });
  if (result.isError) {
    throw new Error(
      `MCP "${toolName}" error: ${JSON.stringify(result.content)}`
    );
  }
  const text = (result.content as Array<{ type: string; text?: string }>)
    .map((c) => (c.type === "text" ? (c.text ?? "") : ""))
    .join("");
  try {
    return JSON.parse(text) as T;
  } catch {
    return text as unknown as T;
  }
}

/** Read an MCP resource and return the parsed content. */
async function mcpResource<T = unknown>(
  client: Client,
  uri: string
): Promise<T> {
  const result = await client.readResource({ uri });
  const text = (result.contents as Array<{ text?: string }>)
    .map((c) => c.text ?? "")
    .join("");
  try {
    return JSON.parse(text) as T;
  } catch {
    return text as unknown as T;
  }
}

// ─── Wallet ───────────────────────────────────────────────────────────────────

function initWallet(): {
  provider: ethers.JsonRpcProvider;
  signer: ethers.Wallet;
} {
  if (!CONFIG.wallet.privateKey)
    throw new Error("AGENT_PRIVATE_KEY is not set.");
  const provider = new ethers.JsonRpcProvider(CONFIG.wallet.rpcUrl);
  const signer = new ethers.Wallet(CONFIG.wallet.privateKey, provider);
  log("INFO", `Investor wallet: ${signer.address}`);
  return { provider, signer };
}

/**
 * Sign and broadcast the unsigned tx payloads returned by purchase_shares.
 *
 * The MCP server constructs the calldata for both:
 *   1. USDC.approve(AssetShares, cost)
 *   2. AssetShares.purchaseShares(assetId, shareCount)
 *
 * The investor's wallet pays for gas (ETH) and the share purchase (USDC).
 * The MCP server never touches the private key.
 */
async function signAndBroadcast(
  signer: ethers.Wallet,
  payloads: UnsignedTx[]
): Promise<string[]> {
  const hashes: string[] = [];
  for (const [i, payload] of payloads.entries()) {
    log("INFO", `  Signing tx ${i + 1}/${payloads.length}  ->  ${payload.to}`);
    const tx = await signer.sendTransaction({
      to: payload.to,
      data: payload.data,
      value: payload.value ? BigInt(payload.value) : 0n,
      gasLimit: 400_000,
    });
    log("INFO", `  Submitted: ${tx.hash}`);
    const receipt = await tx.wait();
    log(
      "INFO",
      `  Confirmed  block=${receipt?.blockNumber}  gas=${receipt?.gasUsed}`
    );
    hashes.push(tx.hash);
  }
  return hashes;
}

// ─── Phase 1 — Criteria Discovery ────────────────────────────────────────────

const CRITERIA_SYSTEM = `
You are a Brickbase investment advisor helping an investor define their
purchase criteria for tokenised commercial real estate shares on Ethereum.

Conduct a focused, natural conversation to learn:
- Target annual yield / return (percentage)
- Maximum price per share (USDC)
- Preferred property locations or types
- Maximum shares per property
- Risk tolerance: low, medium, or high
- Any special conditions (e.g. minimum occupancy, avoid certain sectors)

Once you are confident you have enough information — typically 2-4 exchanges
— embed the criteria in your reply using EXACTLY this fence format:

\`\`\`criteria
{
  "summary": "<one paragraph plain-English summary of all criteria>",
  "minYieldPercent": <number or null>,
  "maxPricePerShareUsdc": <number or null>,
  "preferredLocations": ["<city>"] or null,
  "maxSharesPerProperty": <number or null>,
  "defaultShareCount": <integer - how many shares to buy when criteria pass>,
  "riskTolerance": "low" | "medium" | "high",
  "additionalConditions": ["<free text condition>"]
}
\`\`\`

Do NOT emit the criteria block until you have enough information.
Ask follow-up questions if any critical dimension is unclear.
`.trim();

/**
 * Phase 1: Interactive terminal conversation with Gemini.
 * Continues until the model embeds a criteria block in its reply.
 */
async function discoverCriteria(genAI: GoogleGenAI): Promise<InvestmentCriteria> {
  log("STEP", "=== Phase 1: Investment Criteria Discovery ===");

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const history: Content[] = [];
  let criteria: InvestmentCriteria | null = null;

  const opening =
    "Hello! I am your Brickbase investment advisor.\n\n" +
    "I will help you purchase shares in tokenised commercial properties " +
    "that match your investment goals. Before reviewing the available " +
    "properties, I need to understand your criteria.\n\n" +
    "To start: what annual yield are you targeting, and do you have any " +
    "location or property-type preferences?";

  console.log(`\nAdvisor: ${opening}\n`);
  history.push({ role: "model", parts: [{ text: opening }] });

  while (!criteria) {
    const userInput = await rl.question("You: ");
    if (!userInput.trim()) continue;

    history.push({ role: "user", parts: [{ text: userInput }] });

    const reply = await geminiChat(genAI, history, CRITERIA_SYSTEM);
    history.push({ role: "model", parts: [{ text: reply }] });

    // Check if the model has embedded the criteria JSON block
    const match = reply.match(/```criteria\s*([\s\S]*?)```/);
    const captured = match?.[1];
    if (captured !== undefined) {
      try {
        criteria = JSON.parse(captured.trim()) as InvestmentCriteria;
        // Show the conversational part without the raw JSON block
        const display = reply.replace(/```criteria[\s\S]*?```/, "").trim();
        if (display) console.log(`\nAdvisor: ${display}\n`);
        log("INFO", "Investment criteria captured");
        break;
      } catch {
        // JSON parse failed — keep conversing
        console.log(`\nAdvisor: ${reply}\n`);
      }
    } else {
      console.log(`\nAdvisor: ${reply}\n`);
    }
  }

  rl.close();
  if (!criteria) throw new Error("Failed to extract investment criteria.");
  return criteria;
}

// ─── Phase 2 — Fetch All Properties ──────────────────────────────────────────

async function fetchAllProperties(
  mcpClient: Client
): Promise<PropertyDetail[]> {
  log("STEP", "=== Phase 2: Fetching All AssetVault Properties ===");

  const list = await mcpTool<PropertyDetail[]>(mcpClient, "get_property_list");
  log("INFO", `${list.length} properties found in AssetVault`);

  const details: PropertyDetail[] = [];
  for (const prop of list) {
    try {
      const detail = await mcpTool<PropertyDetail>(
        mcpClient,
        "get_property_detail",
        { assetId: prop.id }
      );
      details.push(detail);
      log("INFO", `  Fetched: ${detail.name ?? prop.id}`);
    } catch (err) {
      log(
        "WARN",
        `  Could not fetch detail for ${prop.id}: ${(err as Error).message}`
      );
      details.push(prop); // fall back to list-level data
    }
    await sleep(200);
  }
  return details;
}

// ─── Phase 3 — Evaluation ────────────────────────────────────────────────────

const EVALUATOR_SYSTEM = `
You are a strict investment evaluator for the Brickbase platform.
Evaluate tokenised commercial real estate properties against an investor's
defined criteria. Return ONLY valid JSON — no markdown, no prose, no fences.
Cite specific numeric values from the property data in your reasoning.
If a required field is absent from the property data, note it and REJECT.
`.trim();

async function evaluateProperty(
  genAI: GoogleGenAI,
  property: PropertyDetail,
  criteria: InvestmentCriteria,
  oraclePrices: Record<string, unknown>
): Promise<EvaluationResult> {
  const prompt = `
Evaluate this Brickbase property against the investor's criteria.

INVESTOR CRITERIA:
${JSON.stringify(criteria, null, 2)}

LIVE ORACLE PRICES (for context):
${JSON.stringify(oraclePrices, null, 2)}

PROPERTY DATA:
${JSON.stringify(property, null, 2)}

Decision rules:
- PURCHASE: shareCount = shares to buy. Respect maxSharesPerProperty and
  availableShares. Use defaultShareCount when no specific cap applies.
- REJECT: shareCount = 0.
- metricsAssessed must map each evaluated criterion to the actual property value.
- If any hard criterion is not met (e.g. yield below minimum), REJECT.

Return ONLY this JSON (no wrapper, no fences, no extra keys):
{
  "decision": "PURCHASE" | "REJECT",
  "shareCount": <integer >= 0>,
  "reasoning": "<2-3 sentences citing specific figures from the property data>",
  "metricsAssessed": { "<criterion name>": "<actual value from data>" }
}
`.trim();

  return geminiJSON<EvaluationResult>(genAI, prompt, EVALUATOR_SYSTEM);
}

// ─── Phase 4 — Execution ─────────────────────────────────────────────────────

async function executePurchase(
  mcpClient: Client,
  signer: ethers.Wallet,
  property: PropertyDetail,
  shareCount: number
): Promise<{ txHashes: string[]; error?: string }> {
  let payload: PurchasePayload;

  try {
    payload = await mcpTool<PurchasePayload>(mcpClient, "purchase_shares", {
      assetId: property.id,
      shareCount,
      buyerAddress: signer.address,
    });
  } catch (err) {
    return {
      txHashes: [],
      error: `purchase_shares MCP call failed: ${(err as Error).message}`,
    };
  }

  if (!payload?.transactions?.length) {
    return {
      txHashes: [],
      error: "purchase_shares returned no transaction payloads.",
    };
  }

  try {
    const hashes = await signAndBroadcast(signer, payload.transactions);
    return { txHashes: hashes };
  } catch (err) {
    return {
      txHashes: [],
      error: `Broadcast failed: ${(err as Error).message}`,
    };
  }
}

// ─── Phase 5 — Report ────────────────────────────────────────────────────────

function buildMarkdownReport(report: AgentReport): string {
  const {
    investorName, walletAddress, timestamp, criteria,
    oraclePrices, deployments, decisions, summary,
  } = report;

  const purchased = decisions.filter(
    (d) => d.decision === "PURCHASE" && !d.error
  );
  const purchasedFailed = decisions.filter(
    (d) => d.decision === "PURCHASE" && !!d.error
  );
  const rejected = decisions.filter((d) => d.decision === "REJECT");

  let md = "# Brickbase Investment Report\n\n---\n\n";

  // Header table
  md += "| | |\n|---|---|\n";
  md += `| **Investor** | ${investorName} |\n`;
  md += `| **Wallet** | \`${walletAddress}\` |\n`;
  md += `| **Generated** | ${timestamp} |\n`;
  md += `| **Chain ID** | ${deployments.chainId ?? "-"} |\n`;
  md += `| **AssetVault** | \`${deployments.assetVaultAddress ?? "-"}\` |\n`;
  md += `| **AssetShares** | \`${deployments.assetSharesAddress ?? "-"}\` |\n\n`;
  md += "---\n\n";

  // Investment Criteria
  md += "## Investment Criteria\n\n";
  md += `> ${criteria.summary}\n\n`;
  md += "| Parameter | Value |\n|---|---|\n";
  if (criteria.minYieldPercent != null)
    md += `| Minimum yield | **${criteria.minYieldPercent}%** |\n`;
  if (criteria.maxPricePerShareUsdc != null)
    md += `| Max price per share | **${criteria.maxPricePerShareUsdc} USDC** |\n`;
  if (criteria.preferredLocations?.length)
    md += `| Preferred locations | ${criteria.preferredLocations.join(", ")} |\n`;
  if (criteria.maxSharesPerProperty != null)
    md += `| Max shares per property | ${criteria.maxSharesPerProperty} |\n`;
  md += `| Default share count | ${criteria.defaultShareCount} |\n`;
  md += `| Risk tolerance | ${criteria.riskTolerance} |\n`;
  if (criteria.additionalConditions.length) {
    md += "\n**Additional conditions:**\n\n";
    for (const c of criteria.additionalConditions) md += `- ${c}\n`;
  }
  md += "\n";

  // Oracle Prices
  md += "## Oracle Prices at Time of Execution\n\n";
  md += "```json\n" + JSON.stringify(oraclePrices, null, 2) + "\n```\n\n";

  // Executive Summary
  const isPlanned = report.planned ?? false;
  md += "## Executive Summary\n\n";
  if (isPlanned) md += "> **Planned decisions** — report written before execution.\n\n";
  md += "| Metric | Value |\n|---|---|\n";
  md += `| Total properties reviewed | **${summary.totalProperties}** |\n`;
  md += `| ${isPlanned ? "Planned" : "Successfully"} purchased shares in | **${summary.purchased}** properties |\n`;
  md += `| Purchase attempts failed | **${purchasedFailed.length}** |\n`;
  md += `| Rejected | **${summary.rejected}** properties |\n`;
  md += `| Total shares ${isPlanned ? "planned" : "acquired"} | **${summary.totalSharesAcquired}** |\n`;
  md += `| Total USDC ${isPlanned ? "planned" : "deployed"} | **${summary.totalUsdcSpent} USDC** |\n\n`;

  // Successful Purchases
  if (purchased.length) {
    md += `## ${isPlanned ? "Planned" : "Purchased"} Properties (${purchased.length})\n\n`;
    for (const d of purchased) {
      md += `### ${d.property.name ?? "Property " + d.property.id}\n\n`;
      md += "| Field | Value |\n|---|---|\n";
      md += `| Asset ID | \`${d.property.id}\` |\n`;
      md += `| Location | ${d.property.location ?? "-"} |\n`;
      md += `| Shares purchased | **${d.shareCount}** |\n`;
      md += `| Price per share | ${d.pricePerShare} USDC |\n`;
      md += `| Total cost | **${d.totalCostUsdc} USDC** |\n`;
      for (const tx of d.txHashes)
        md += `| Transaction | \`${tx}\` |\n`;
      md += `\n**Rationale:** ${d.reasoning}\n\n`;
      if (Object.keys(d.metricsAssessed).length) {
        md += "**Metrics assessed:**\n\n| Criterion | Value found |\n|---|---|\n";
        for (const [k, v] of Object.entries(d.metricsAssessed))
          md += `| ${k} | ${v} |\n`;
      }
      md += "\n---\n\n";
    }
  }

  // Failed Purchases
  if (purchasedFailed.length) {
    md += `## Purchase Attempts Failed (${purchasedFailed.length})\n\n`;
    for (const d of purchasedFailed) {
      md += `### ${d.property.name ?? "Property " + d.property.id}\n\n`;
      md += "| Field | Value |\n|---|---|\n";
      md += `| Asset ID | \`${d.property.id}\` |\n`;
      md += `| Shares attempted | ${d.shareCount} |\n`;
      md += `\n**Gemini recommendation:** ${d.reasoning}\n\n`;
      md += `> **Execution error:** ${d.error}\n\n`;
      md += "---\n\n";
    }
  }

  // Rejections
  if (rejected.length) {
    md += `## Rejected Properties (${rejected.length})\n\n`;
    for (const d of rejected) {
      md += `### ${d.property.name ?? "Property " + d.property.id}\n\n`;
      md += "| Field | Value |\n|---|---|\n";
      md += `| Asset ID | \`${d.property.id}\` |\n`;
      md += `| Location | ${d.property.location ?? "-"} |\n`;
      md += `| Price per share | ${d.pricePerShare} USDC |\n`;
      if (Object.keys(d.metricsAssessed ?? {}).length) {
        for (const [k, v] of Object.entries(d.metricsAssessed))
          md += `| ${k} | ${v} |\n`;
      }
      md += `\n**Reason for rejection:** ${d.reasoning}\n\n`;
      md += "---\n\n";
    }
  }

  return md;
}

async function writeReport(report: AgentReport): Promise<string> {
  await fs.mkdir(CONFIG.reportDir, { recursive: true });
  const slug = new Date()
    .toISOString()
    .replace(/:/g, "-")
    .replace(/\..+/, "");
  const filePath = path.join(CONFIG.reportDir, `report-${slug}.md`);
  await fs.writeFile(filePath, buildMarkdownReport(report), "utf8");
  return filePath;
}

// ─── Main ─────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  console.log("\n" + "=".repeat(70));
  console.log("  Brickbase Investor Agent");
  console.log("  LLM  : Google Gemini  (@google/genai - centralised SDK)");
  console.log("  Data : brickbase MCP server (tools + config://deployments)");
  console.log("  Chain: investor wallet signs MCP-generated unsigned tx payloads");
  console.log("=".repeat(70) + "\n");

  // ── Initialise all subsystems ─────────────────────────────────────────────
  const genAI = initGemini();
  const { signer, provider } = initWallet();
  const mcpClient = await initMCP();

  // ── Resolve contract addresses from the MCP config resource ──────────────
  //    The agent does NOT need ASSET_SHARES_ADDRESS as an env var.
  //    config://deployments is the single source of truth.
  const deployments = await mcpResource<Record<string, unknown>>(
    mcpClient,
    "config://deployments"
  );
  log("INFO", `Chain ID    : ${deployments.chainId}`);
  log("INFO", `AssetVault  : ${deployments.assetVaultAddress ?? "-"}`);
  log("INFO", `AssetShares : ${deployments.assetSharesAddress ?? "-"}`);

  // ── Whitelist check ───────────────────────────────────────────────────────
  try {
    const wl = await mcpTool<{ whitelisted: boolean }>(
      mcpClient,
      "get_user_whitelist_status",
      { address: signer.address }
    );
    if (!wl.whitelisted) {
      log("WARN", `Wallet ${signer.address} is NOT whitelisted.`);
      log(
        "WARN",
        "Share purchases will likely revert. Ask an admin to whitelist this wallet first."
      );
    } else {
      log("INFO", `Wallet ${signer.address} is whitelisted`);
    }
  } catch (err) {
    log("WARN", `Whitelist check failed: ${(err as Error).message}`);
  }

  // ── Fetch oracle prices ───────────────────────────────────────────────────
  const oraclePrices = await mcpTool<Record<string, unknown>>(
    mcpClient,
    "get_oracle_prices"
  );
  log("INFO", `Oracle prices fetched: ${Object.keys(oraclePrices).join(", ")}`);

  // ── Phase 1: Criteria Discovery (interactive) ─────────────────────────────
  const criteria = await discoverCriteria(genAI);

  // ── Phase 2: Fetch all properties ─────────────────────────────────────────
  const properties = await fetchAllProperties(mcpClient);

  // ── Phase 3: Evaluate all properties (no purchases yet) ───────────────────
  log("STEP", `=== Phase 3: Evaluating ${properties.length} Properties ===`);

  const decisions: PropertyDecision[] = [];
  let plannedSharesTotal = 0;
  let plannedUsdcTotal = 0;

  for (const property of properties) {
    const label = property.name ?? `Property ${property.id}`;
    log("INFO", `Evaluating: ${label}`);

    let evaluation: EvaluationResult;
    try {
      evaluation = await evaluateProperty(
        genAI,
        property,
        criteria,
        oraclePrices
      );
    } catch (err) {
      log("WARN", `  Evaluation error: ${(err as Error).message}`);
      decisions.push({
        property,
        decision: "REJECT",
        shareCount: 0,
        pricePerShare: String(property.pricePerShare ?? "unknown"),
        totalCostUsdc: "0",
        reasoning: `Evaluation failed: ${(err as Error).message}`,
        metricsAssessed: {},
        txHashes: [],
      });
      continue;
    }

    const pricePerShare = String(property.pricePerShare ?? "0");
    const priceNum = parseFloat(pricePerShare) || 0;
    const costNum = priceNum * evaluation.shareCount;
    const totalCostUsdc = costNum.toFixed(2);

    if (evaluation.decision === "PURCHASE" && evaluation.shareCount > 0) {
      log(
        "STEP",
        `  PURCHASE: ${evaluation.shareCount} shares  (${totalCostUsdc} USDC)`
      );
      plannedSharesTotal += evaluation.shareCount;
      plannedUsdcTotal += costNum;
      decisions.push({
        property,
        decision: "PURCHASE",
        shareCount: evaluation.shareCount,
        pricePerShare,
        totalCostUsdc,
        reasoning: evaluation.reasoning,
        metricsAssessed: evaluation.metricsAssessed,
        txHashes: [],
      });
    } else {
      log("INFO", `  REJECT: ${evaluation.reasoning.slice(0, 90)}...`);
      decisions.push({
        property,
        decision: "REJECT",
        shareCount: 0,
        pricePerShare,
        totalCostUsdc: "0",
        reasoning: evaluation.reasoning,
        metricsAssessed: evaluation.metricsAssessed,
        txHashes: [],
      });
    }

    await sleep(700); // rate-limit guard between Gemini calls
  }

  // ── Phase 4: Write report (planned decisions, before any purchase) ─────────
  log("STEP", "=== Phase 4: Writing Investment Report (before execution) ===");

  const plannedPurchases = decisions.filter((d) => d.decision === "PURCHASE");
  const report: AgentReport = {
    investorName: CONFIG.investorName,
    walletAddress: signer.address,
    timestamp: new Date().toISOString(),
    criteria,
    oraclePrices,
    deployments,
    decisions,
    planned: true,
    summary: {
      totalProperties: properties.length,
      purchased: plannedPurchases.length,
      rejected: decisions.length - plannedPurchases.length,
      totalSharesAcquired: plannedSharesTotal,
      totalUsdcSpent: plannedUsdcTotal.toFixed(2),
    },
  };

  const reportPath = await writeReport(report);

  // Console summary (before execution)
  console.log("\n" + "=".repeat(70));
  console.log("  PLANNED DECISIONS (before execution)");
  console.log("=".repeat(70));
  console.log(`  Properties reviewed   : ${report.summary.totalProperties}`);
  console.log(`  Planned purchases     : ${report.summary.purchased}`);
  console.log(`  Rejected              : ${report.summary.rejected}`);
  console.log(`  Planned shares        : ${report.summary.totalSharesAcquired}`);
  console.log(`  Planned USDC         : ${report.summary.totalUsdcSpent}`);
  console.log(`  Report written        : ${reportPath}`);
  console.log("=".repeat(70) + "\n");

  // ── Phase 5: Execute purchases ───────────────────────────────────────────
  log("STEP", "=== Phase 5: Executing Purchases ===");

  let totalSharesAcquired = 0;
  let totalUsdcSpentNum = 0;

  for (const d of decisions) {
    if (d.decision !== "PURCHASE" || d.shareCount === 0) continue;

    const label = d.property.name ?? `Property ${d.property.id}`;
    log("INFO", `Executing: ${label} (${d.shareCount} shares)`);

    const { txHashes, error } = await executePurchase(
      mcpClient,
      signer,
      d.property,
      d.shareCount
    );

    d.txHashes = txHashes;
    if (error !== undefined) Object.assign(d, { error });

    if (!error) {
      totalSharesAcquired += d.shareCount;
      totalUsdcSpentNum += parseFloat(d.totalCostUsdc) || 0;
      log("INFO", `  Purchased. Tx(s): ${txHashes.join(", ")}`);
    } else {
      log("ERROR", `  Purchase failed: ${error}`);
    }
  }

  // Update report with execution results
  const successfulPurchases = decisions.filter(
    (x) => x.decision === "PURCHASE" && !x.error
  ).length;
  report.planned = false;
  report.summary = {
    totalProperties: properties.length,
    purchased: successfulPurchases,
    rejected: decisions.filter((x) => x.decision === "REJECT").length,
    totalSharesAcquired,
    totalUsdcSpent: totalUsdcSpentNum.toFixed(2),
  };
  await writeReport(report);

  console.log("\n" + "=".repeat(70));
  console.log("  EXECUTION COMPLETE");
  console.log("=".repeat(70));
  console.log(`  Successfully purchased: ${report.summary.purchased}`);
  console.log(`  Total shares acquired : ${report.summary.totalSharesAcquired}`);
  console.log(`  Total USDC spent      : ${report.summary.totalUsdcSpent}`);
  console.log(`  Report updated        : ${reportPath}`);
  console.log("=".repeat(70) + "\n");

  await (mcpClient as Client & { close?: () => Promise<void> }).close?.();
  provider.destroy?.();
}

// ─── Graceful shutdown ────────────────────────────────────────────────────────

process.on("SIGINT", () => {
  log("INFO", "Interrupted - exiting.");
  process.exit(0);
});

main().catch((err: unknown) => {
  console.error("Fatal:", err);
  process.exit(1);
});
