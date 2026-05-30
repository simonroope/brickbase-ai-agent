/**
 * ============================================================
 * RWA Investor Agent
 * ============================================================
 *
 * An agent for investors with no affiliation to any platform.
 * After defining investment criteria, the agent finds a compatible
 * platform, connects via MCP, discovers its tools and resources,
 * and executes the investment strategy.
 *
 *  1. CRITERIA DISCOVERY
 *     Multi-turn conversation to elicit purchase conditions:
 *     yield target, max price, locations, risk appetite, etc.
 *     Also captures the platform URL (or uses MCP_URL).
 *
 *  2. PLATFORM CONNECTION
 *     Connects to the MCP server at the provided URL.
 *     Discovers available tools and resources (no prior knowledge).
 *
 *  3. CAPABILITY MAPPING
 *     Uses the LLM to map discovered tools/resources to the
 *     agent's needs (fetch assets, fetch detail, purchase, config).
 *
 *  4. EVALUATION
 *     Fetches assets, evaluates against criteria.
 *
 *  5. REPORT
 *     Writes a Markdown report of planned decisions (before any purchase).
 *
 *  6. EXECUTION
 *     Executes purchases via the discovered tools, signs with wallet.
 *
 * ── Environment variables ──────────────────────────────────────
 *   GOOGLE_API_KEY      Gemini API key
 *   GEMINI_MODEL        Gemini model name (default: gemini-2.0-flash)
 *   AGENT_PRIVATE_KEY   Investor wallet private key (0x-prefixed)
 *   MCP_URL             Platform MCP URL (optional; can be provided during criteria discovery)
 *   RPC_URL             Ethereum RPC (default: http://localhost:8545)
 *   REPORT_DIR          Output directory (default: ./reports)
 * ============================================================
 */

import * as fs from "fs/promises";
import * as path from "path";
import * as readline from "readline/promises";
import { pathToFileURL } from "node:url";
import type { GoogleGenAI, Content } from "@google/genai";
import type { Client } from "@modelcontextprotocol/sdk/client/index.js";
import type { JsonRpcProvider, Wallet } from "ethers";

/** Heavy deps are loaded lazily so startup logs appear before any import can hang. */
let googleGenaiModule: typeof import("@google/genai") | undefined;
let mcpClientModule: typeof import("@modelcontextprotocol/sdk/client/index.js") | undefined;
let mcpTransportModule:
  | typeof import("@modelcontextprotocol/sdk/client/streamableHttp.js")
  | undefined;
let ethersModule: typeof import("ethers") | undefined;

console.log(
  `[${new Date().toISOString()}] [>]  Agent module loaded — waiting for main()`
);

// ─── Types ────────────────────────────────────────────────────────────────────

interface InvestmentCriteria {
  summary: string;
  minYieldPercent?: number;
  maxPricePerShareUsdc?: number;
  preferredLocations?: string[];
  maxSharesPerAsset?: number;
  defaultShareCount: number;
  riskTolerance: "low" | "medium" | "high";
  additionalConditions: string[];
  platformUrl?: string;
}

interface McpTool {
  name: string;
  description?: string;
  inputSchema?: Record<string, unknown>;
}

interface McpResource {
  uri: string;
  name?: string;
  description?: string;
}

interface DiscoveredCapabilities {
  fetchListTool: string;
  fetchDetailTool: string;
  detailIdParam: string;
  purchaseTool: string;
  purchaseAssetParam: string;
  purchaseAmountParam: string;
  configResource?: string;
}

interface AssetDetail {
  assetId?: number;
  name: string;
  [key: string]: unknown;
}

function getAssetId(asset: Record<string, unknown>): number | undefined {
  const assetId = asset.assetId;
  if (assetId == null) return undefined;
  const n = Number(assetId);
  return Number.isNaN(n) ? undefined : n;
}

function assetDisplayName(asset: AssetDetail): string {
  const a = asset as Record<string, unknown>;
  const name = a.name ?? (a.metadata as Record<string, unknown> | undefined)?.name;
  if (typeof name === "string" && name) return name;
  const assetId = getAssetId(a);
  return assetId != null ? `Asset ${assetId}` : "Unknown Asset";
}

interface EvaluationResult {
  decision: "PURCHASE" | "REJECT";
  shareCount: number;
  reasoning: string;
  metricsAssessed: Record<string, string>;
}

interface AssetDecision {
  asset: AssetDetail;
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
  walletAddress: string;
  timestamp: string;
  criteria: InvestmentCriteria;
  platformUrl: string;
  decisions: AssetDecision[];
  planned?: boolean;
  summary: {
    totalAssets: number;
    purchased: number;
    rejected: number;
    totalSharesAcquired: number;
    totalUsdcSpent: string;
  };
}

// ─── Config ───────────────────────────────────────────────────────────────────

/** USDC and share prices use 6 decimals on-chain */
const USDC_DECIMALS = 6;

function getConfig() {
  return {
    gemini: {
      model: process.env.GEMINI_MODEL ?? "gemini-2.0-flash",
      maxOutputTokens: 2048,
      temperature: 0.4,
    },
    mcp: {
      url: process.env.MCP_URL ?? "",
    },
    wallet: {
      privateKey: process.env.AGENT_PRIVATE_KEY ?? "",
      rpcUrl: process.env.RPC_URL ?? "http://localhost:8545",
    },
    reportDir: process.env.REPORT_DIR ?? "./reports",
  };
}

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

async function waitMs(reason: string, ms: number): Promise<void> {
  log("INFO", `Waiting ${ms}ms — ${reason}`);
  await sleep(ms);
}

async function loadDependencies(): Promise<void> {
  log("INFO", "Loading dotenv...");
  const dotenv = await import("dotenv");
  dotenv.config();

  log("INFO", "Loading @google/genai...");
  googleGenaiModule = await import("@google/genai");

  log("INFO", "Loading MCP SDK...");
  mcpClientModule = await import("@modelcontextprotocol/sdk/client/index.js");
  mcpTransportModule = await import(
    "@modelcontextprotocol/sdk/client/streamableHttp.js"
  );

  log("INFO", "Loading ethers...");
  ethersModule = await import("ethers");

  log("INFO", "Dependencies loaded");
}

// ─── Gemini ───────────────────────────────────────────────────────────────────

function initGemini(): GoogleGenAI {
  if (!googleGenaiModule) {
    throw new Error("Dependencies not loaded. Call loadDependencies() first.");
  }
  const apiKey = process.env.GOOGLE_API_KEY;
  if (!apiKey) throw new Error("GOOGLE_API_KEY is not set.");
  const config = getConfig();
  log("INFO", `Gemini AI client ready (model: ${config.gemini.model})`);
  return new googleGenaiModule.GoogleGenAI({ apiKey });
}

async function geminiChat(
  genAI: GoogleGenAI,
  history: Content[],
  systemInstruction: string
): Promise<string> {
  const config = getConfig();
  const response = await genAI.models.generateContent({
    model: config.gemini.model,
    contents: history,
    config: {
      maxOutputTokens: config.gemini.maxOutputTokens,
      temperature: config.gemini.temperature,
      systemInstruction,
    },
  });
  return response.text ?? "";
}

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

async function connectMCP(url: string): Promise<Client> {
  if (!mcpClientModule || !mcpTransportModule) {
    throw new Error("Dependencies not loaded. Call loadDependencies() first.");
  }
  const { Client: McpClient } = mcpClientModule;
  const { StreamableHTTPClientTransport } = mcpTransportModule;
  const client = new McpClient({
    name: "rwa-investor-agent",
    version: "1.0.0",
  });
  try {
    await client.connect(
      new StreamableHTTPClientTransport(new URL(url)) as Parameters<
        Client["connect"]
      >[0]
    );
  } catch (err) {
    const code = (err as { code?: number })?.code;
    const msg =
      code === 404
        ? `Platform not found at ${url}.`
        : code && code >= 500
          ? `Platform at ${url} returned error ${code}.`
          : `Platform is not reachable at ${url}.`;
    throw new Error(msg, { cause: err });
  }
  log("INFO", `Connected to platform: ${url}`);
  return client;
}

async function mcpTool<T = unknown>(
  client: Client,
  toolName: string,
  args: Record<string, unknown> = {}
): Promise<T> {
  const result = await client.callTool({ name: toolName, arguments: args });
  if (result.isError) {
    throw new Error(
      `Tool "${toolName}" error: ${JSON.stringify(result.content)}`
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

async function discoverCapabilities(
  genAI: GoogleGenAI,
  client: Client
): Promise<DiscoveredCapabilities> {
  const toolsResult = await client.listTools();
  const resourcesResult = await client.listResources();

  const tools: McpTool[] = (toolsResult.tools ?? []).map((t) => {
    const tool: McpTool = { name: t.name };
    if (t.description !== undefined) tool.description = t.description;
    if (t.inputSchema) tool.inputSchema = t.inputSchema as Record<string, unknown>;
    return tool;
  });

  const resources: McpResource[] = (resourcesResult.resources ?? []).map(
    (r) => {
      const res: McpResource = { uri: r.uri };
      if (r.name !== undefined) res.name = r.name;
      if (r.description !== undefined) res.description = r.description;
      return res;
    }
  );

  log("INFO", `Discovered ${tools.length} tools, ${resources.length} resources`);
  log("INFO", `Tools: ${tools.map((t) => t.name).join(", ")}`);
  log("INFO", `Resources: ${resources.map((r) => r.uri).join(", ")}`);

  const prompt = `
You are mapping MCP tools and resources for an RWA investment agent.
The agent needs to: (1) fetch a list of investable assets, (2) fetch detail for one asset by ID,
(3) prepare purchase transactions for an asset, (4) optionally read chain config.

TOOLS:
${JSON.stringify(tools, null, 2)}

RESOURCES:
${JSON.stringify(resources, null, 2)}

Return JSON with the exact names/URIs to use. Infer parameter names from inputSchema.
{
  "fetchListTool": "<tool name that fetches list of assets>",
  "fetchDetailTool": "<tool name that fetches one asset by ID>",
  "detailIdParam": "<parameter name for asset ID, e.g. assetId>",
  "purchaseTool": "<tool name that prepares purchase transactions>",
  "purchaseAssetParam": "<parameter name for asset ID>",
  "purchaseAmountParam": "<parameter name for share amount>",
  "configResource": "<resource URI for chain config, or null>"
}
`.trim();

  const caps = await geminiJSON<DiscoveredCapabilities>(
    genAI,
    prompt,
    "Return only valid JSON. Use exact tool names and parameter names from the schemas."
  );
  log("INFO", `DiscoveredCapabilities: ${JSON.stringify(caps, null, 2)}`);
  return caps;
}

// ─── Wallet ───────────────────────────────────────────────────────────────────

function initWallet(): {
  provider: JsonRpcProvider;
  signer: Wallet;
} {
  if (!ethersModule) {
    throw new Error("Dependencies not loaded. Call loadDependencies() first.");
  }
  const config = getConfig();
  if (!config.wallet.privateKey)
    throw new Error("AGENT_PRIVATE_KEY is not set.");
  const provider = new ethersModule.ethers.JsonRpcProvider(config.wallet.rpcUrl);
  const signer = new ethersModule.ethers.Wallet(config.wallet.privateKey, provider);
  log("INFO", `Investor wallet: ${signer.address}`);
  return { provider, signer };
}

async function signAndBroadcast(
  signer: Wallet,
  payloads: Array<{ to: string; data: string; value?: string }>
): Promise<string[]> {
  const hashes: string[] = [];
  for (const [i, p] of payloads.entries()) {
    await waitMs("nonce propagation before signing next tx", 2000);
    const nonce = await signer.getNonce("pending");
    log("INFO", `  Signing tx ${i + 1}/${payloads.length}  ->  ${p.to}`);
    const tx = await signer.sendTransaction({
      to: p.to,
      data: p.data,
      value: p.value ? BigInt(p.value) : 0n,
      gasLimit: 400_000,
      nonce,
    });
    log("INFO", `  Submitted: ${tx.hash}`);
    log("INFO", `  Waiting for tx ${tx.hash} to be mined...`);
    const receipt = await tx.wait();
    if (receipt) {
      log("INFO", `  Receipt: blockNumber=${receipt.blockNumber} gasUsed=${receipt.gasUsed}`);
    }
    hashes.push(tx.hash);
  }
  return hashes;
}

// ─── Phase 1 — Criteria Discovery ──────────────────────────────────────────────

const CRITERIA_SYSTEM = `
You are an investment advisor helping an investor define criteria for
tokenised real estate (RWA) shares. You have no affiliation to any platform.

Conduct a focused conversation to learn:
- Target annual yield
- Maximum price per share (USDC)
- Preferred locations or asset types
- Maximum shares per asset
- Risk tolerance: low, medium, or high
- Any special conditions

At the end, ask: "Which platform would you like to use? (Enter the URL, e.g. https://example.com)"

Once you have enough information AND the platform URL, embed the criteria in this format:

\`\`\`criteria
{
  "summary": "<one paragraph summary of all criteria>",
  "minYieldPercent": <number or null>,
  "maxPricePerShareUsdc": <number or null>,
  "preferredLocations": ["<city>"] or null,
  "maxSharesPerAsset": <number or null>,
  "defaultShareCount": <integer>,
  "riskTolerance": "low" | "medium" | "high",
  "additionalConditions": ["<condition>"],
  "platformUrl": "<URL the user provided>"
}
\`\`\`

Do NOT emit the criteria block until you have the platform URL.
`.trim();

async function discoverCriteria(
  genAI: GoogleGenAI
): Promise<InvestmentCriteria> {
  log("STEP", "=== Phase 1: Investment Criteria Discovery ===");

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const history: Content[] = [];
  let criteria: InvestmentCriteria | null = null;

  const opening =
    "Hello! I am your RWA investment advisor.\n\n" +
    "I will help you invest in tokenised real estate that matches your goals. " +
    "I have no affiliation to any platform — we will find one together.\n\n" +
    "To start: what annual yield are you targeting, and any location preferences?";

  console.log(`\nAdvisor: ${opening}\n`);
  history.push({ role: "model", parts: [{ text: opening }] });

  while (!criteria) {
    log("INFO", "Waiting for your input...");
    const userInput = await rl.question("You: ");
    if (!userInput.trim()) continue;

    history.push({ role: "user", parts: [{ text: userInput }] });

    log("INFO", "Waiting for Gemini (criteria discovery)...");
    const reply = await geminiChat(genAI, history, CRITERIA_SYSTEM);
    history.push({ role: "model", parts: [{ text: reply }] });

    const match = reply.match(/```criteria\s*([\s\S]*?)```/);
    const captured = match?.[1];
    if (captured !== undefined) {
      try {
        criteria = JSON.parse(captured.trim()) as InvestmentCriteria;
        const display = reply.replace(/```criteria[\s\S]*?```/, "").trim();
        if (display) console.log(`\nAdvisor: ${display}\n`);
        log("INFO", "Investment criteria captured");
        break;
      } catch {
        console.log(`\nAdvisor: ${reply}\n`);
      }
    } else {
      console.log(`\nAdvisor: ${reply}\n`);
    }
  }

  rl.close();
  if (!criteria) throw new Error("Failed to extract investment criteria.");

  criteria.platformUrl =
    criteria.platformUrl ?? getConfig().mcp.url;
  if (!criteria.platformUrl)
    throw new Error(
      "Platform URL is required. Set MCP_URL or provide it during the conversation."
    );

  return criteria;
}

// ─── Phase 2 — Fetch Assets (discovery-based) ──────────────────────────────────

/** Convert on-chain e6 value (string or number) to human-readable USDC */
function fromUsdcUnits(raw: unknown): number {
  if (raw == null) return 0;
  const s = String(raw).trim();
  if (!s) return 0;
  const n = parseFloat(s);
  return Number.isNaN(n) ? 0 : n / 10 ** USDC_DECIMALS;
}

/** Normalize asset for evaluation: add human-readable price/value fields (USDC) */
function normalizeAssetForEvaluation(asset: Record<string, unknown>): Record<string, unknown> {
  const normalized = { ...asset };
  const sharePrice = asset.sharePrice ?? asset.pricePerShare;
  normalized.pricePerShareUsdc = fromUsdcUnits(sharePrice);
  normalized.capitalValueUsdc = fromUsdcUnits(asset.capitalValue);
  normalized.incomeValueUsdc = fromUsdcUnits(asset.incomeValue);
  normalized.totalSupplyUsdc = fromUsdcUnits(asset.totalSupply);
  normalized.availableSupplyUsdc = fromUsdcUnits(asset.availableSupply);
  return normalized;
}

function extractAssetList(list: unknown): unknown[] {
  if (Array.isArray(list)) return list;
  if (list && typeof list === "object") {
    const obj = list as Record<string, unknown>;
    for (const key of ["assets", "data", "items", "results"]) {
      const val = obj[key];
      if (Array.isArray(val)) return val;
    }
  }
  return [list];
}

async function fetchAllAssets(
  client: Client,
  caps: DiscoveredCapabilities
): Promise<AssetDetail[]> {
  log("STEP", "=== Phase 2: Fetching Assets ===");

  const raw = await mcpTool<unknown>(client, caps.fetchListTool, {});
  const items = extractAssetList(raw);
  log("INFO", `${items.length} assets found`);

  const details: AssetDetail[] = [];
  for (const asset of items) {
    const idNumber = getAssetId(asset as Record<string, unknown>);
    if (idNumber == null) {
      log("WARN", `  Skipping item with no assetId: ${JSON.stringify(asset).slice(0, 80)}...`);
      details.push(asset as AssetDetail);
      continue;
    }
    try {
      const detail = await mcpTool<AssetDetail>(
        client,
        caps.fetchDetailTool,
        { [caps.detailIdParam]: idNumber }
      );
      details.push(detail);
      log("INFO", `  Fetched: ${(detail as AssetDetail).name ?? idNumber}`);
    } catch (err) {
      log("WARN", `  Could not fetch detail for ${idNumber}: ${(err as Error).message}`);
      details.push(asset as AssetDetail);
    }
    await waitMs(`rate limit before fetching next asset detail (${idNumber})`, 200);
  }
  return details;
}

// ─── Phase 3 — Evaluation ─────────────────────────────────────────────────────

const EVALUATOR_SYSTEM = `
You are a strict investment evaluator for tokenised real estate.
Evaluate assets against the investor's criteria. Return ONLY valid JSON.
Use the *_Usdc fields for comparisons (pricePerShareUsdc, capitalValueUsdc, incomeValueUsdc).
These are human-readable USDC amounts. Compare pricePerShareUsdc to maxPricePerShareUsdc.
Cite specific numeric values in your reasoning.
`.trim();

async function evaluateAsset(
  genAI: GoogleGenAI,
  asset: AssetDetail,
  criteria: InvestmentCriteria,
  context: Record<string, unknown>
): Promise<EvaluationResult> {
  const normalized = normalizeAssetForEvaluation(asset as Record<string, unknown>);
  const prompt = `
Evaluate this asset against the investor's criteria.

CRITERIA:
${JSON.stringify(criteria, null, 2)}

CONTEXT:
${JSON.stringify(context, null, 2)}

ASSET DATA (pricePerShareUsdc, capitalValueUsdc, incomeValueUsdc are human-readable USDC):
${JSON.stringify(normalized, null, 2)}

Return JSON:
{
  "decision": "PURCHASE" | "REJECT",
  "shareCount": <integer >= 0>,
  "reasoning": "<2-3 sentences citing specific figures>",
  "metricsAssessed": { "<criterion>": "<actual value>" }
}
`.trim();

  return geminiJSON<EvaluationResult>(genAI, prompt, EVALUATOR_SYSTEM);
}

// ─── Phase 4 — Execution ───────────────────────────────────────────────────────

async function executePurchase(
  client: Client,
  signer: Wallet,
  asset: AssetDetail,
  shareCount: number,
  caps: DiscoveredCapabilities
): Promise<{ txHashes: string[]; error?: string }> {
  try {
    const id = getAssetId(asset as Record<string, unknown>);
    if (id == null) throw new Error(`Asset has no assetId: ${JSON.stringify(asset).slice(0, 100)}`);
    const payload = await mcpTool<{
      transactions?: Array<{ to: string; data: string; value?: string }>;
    }>(client, caps.purchaseTool, {
      [caps.purchaseAssetParam]: id,
      [caps.purchaseAmountParam]: String(shareCount),
    });

    const txList = payload?.transactions ?? [];
    if (!txList.length) {
      return { txHashes: [], error: "Purchase tool returned no transactions." };
    }

    const hashes = await signAndBroadcast(signer, txList);
    return { txHashes: hashes };
  } catch (err) {
    return {
      txHashes: [],
      error: `Purchase failed: ${(err as Error).message}`,
    };
  }
}

// ─── Phase 5 — Report ───────────────────────────────────────────────────────────

function buildMarkdownReport(report: AgentReport): string {
  const { walletAddress, timestamp, criteria, decisions, summary } = report;

  let md = "# RWA Investment Report\n\n---\n\n";
  md += "| | |\n|---|---|\n";
  md += `| **Wallet** | \`${walletAddress}\` |\n`;
  md += `| **Generated** | ${timestamp} |\n`;
  md += `| **Platform** | ${report.platformUrl} |\n\n`;
  md += "---\n\n";

  md += "## Investment Criteria\n\n";
  md += `> ${criteria.summary}\n\n`;
  md += "---\n\n";

  md += "## Summary\n\n";
  md += `| Metric | Value |\n|---|---|\n`;
  md += `| Total reviewed | **${summary.totalAssets}** |\n`;
  md += `| Purchased | **${summary.purchased}** |\n`;
  md += `| Rejected | **${summary.rejected}** |\n`;
  md += `| Total shares | **${summary.totalSharesAcquired}** |\n`;
  md += `| Total USDC | **${summary.totalUsdcSpent}** |\n\n`;

  // Decisions grid
  md += "## Decisions\n\n";
  md += "| Asset | Decision | Reason |\n";
  md += "|----------|----------|--------|\n";
  for (const d of decisions) {
    const name = assetDisplayName(d.asset);
    let reason = (d.reasoning + (d.error ? ` (Error: ${d.error})` : "")).replace(/\|/g, "\\|").replace(/\n/g, " ");
    if (reason.length > 120) reason = reason.slice(0, 117) + "...";
    md += `| ${name} | ${d.decision} | ${reason} |\n`;
  }
  md += "\n";

  // Detailed per-asset
  for (const d of decisions) {
    md += `### ${assetDisplayName(d.asset)}\n\n`;
    md += `| Field | Value |\n|---|---|\n`;
    md += `| Decision | ${d.decision} |\n`;
    md += `| Shares | ${d.shareCount} |\n`;
    md += `| Cost | ${d.totalCostUsdc} USDC |\n`;
    md += `\n**Rationale:** ${d.reasoning}\n\n`;
    if (d.txHashes.length) {
      for (const tx of d.txHashes) md += `| Tx | \`${tx}\` |\n`;
    }
    if (d.error) md += `\n> Error: ${d.error}\n`;
    md += "\n---\n\n";
  }

  return md;
}

async function writeReport(report: AgentReport): Promise<string> {
  const reportDir = getConfig().reportDir;
  await fs.mkdir(reportDir, { recursive: true });
  const slug = new Date()
    .toISOString()
    .replace(/:/g, "-")
    .replace(/\..+/, "");
  const filePath = path.join(reportDir, `report-${slug}.md`);
  await fs.writeFile(filePath, buildMarkdownReport(report), "utf8");
  return filePath;
}

// ─── Main ─────────────────────────────────────────────────────────────────────

export async function main(): Promise<void> {
  log("STEP", "Script started — RWA Investor Agent");
  await loadDependencies();

  console.log("\n" + "=".repeat(70));
  console.log("  RWA Investor Agent");
  console.log("  Platform-agnostic • Discovers tools and resources via MCP");
  console.log("=".repeat(70) + "\n");

  const genAI = initGemini();
  const { signer, provider } = initWallet();

  // Phase 1: Criteria (includes platform URL)
  const criteria = await discoverCriteria(genAI);
  const platformUrl = criteria.platformUrl!.replace(/\/$/, "") + "/mcp";

  // Phase 2: Connect and discover
  log("STEP", `=== Connecting to platform at ${platformUrl} ===`);
  const mcpClient = await connectMCP(platformUrl);
  const caps = await discoverCapabilities(genAI, mcpClient);

  // Optional: fetch config for context
  let configContext: Record<string, unknown> = {};
  if (caps.configResource) {
    try {
      configContext = await mcpResource<Record<string, unknown>>(
        mcpClient,
        caps.configResource
      );
    } catch {
      configContext = {};
    }
  }

  // Phase 3: Fetch and evaluate
  const assets = await fetchAllAssets(mcpClient, caps);

  log("STEP", `=== Phase 3: Evaluating ${assets.length} Assets ===`);

  const decisions: AssetDecision[] = [];
  let plannedSharesTotal = 0;
  let plannedUsdcTotal = 0;

  for (const asset of assets) {
    const label = assetDisplayName(asset);
    log("INFO", `Evaluating: ${label}`);

    let evaluation: EvaluationResult;
    try {
      log("INFO", `Waiting for Gemini (evaluate ${label})...`);
      evaluation = await evaluateAsset(
        genAI,
        asset,
        criteria,
        configContext
      );
    } catch (err) {
      log("WARN", `  Evaluation error: ${(err as Error).message}`);
      const fallbackPrice = fromUsdcUnits(
        (asset as Record<string, unknown>).sharePrice ??
          (asset as Record<string, unknown>).pricePerShare
      );
      decisions.push({
        asset,
        decision: "REJECT",
        shareCount: 0,
        pricePerShare: fallbackPrice > 0 ? fallbackPrice.toFixed(2) : "unknown",
        totalCostUsdc: "0",
        reasoning: `Evaluation failed: ${(err as Error).message}`,
        metricsAssessed: {},
        txHashes: [],
      });
      continue;
    }

    const pricePerShareUsdc = fromUsdcUnits(
      (asset as Record<string, unknown>).sharePrice ??
        (asset as Record<string, unknown>).pricePerShare
    );
    const costNum = pricePerShareUsdc * evaluation.shareCount;
    const totalCostUsdc = costNum.toFixed(2);
    const pricePerShare = pricePerShareUsdc.toFixed(2);

    if (evaluation.decision === "PURCHASE" && evaluation.shareCount > 0) {
      plannedSharesTotal += evaluation.shareCount;
      plannedUsdcTotal += costNum;
      decisions.push({
        asset,
        decision: "PURCHASE",
        shareCount: evaluation.shareCount,
        pricePerShare,
        totalCostUsdc,
        reasoning: evaluation.reasoning,
        metricsAssessed: evaluation.metricsAssessed,
        txHashes: [],
      });
    } else {
      decisions.push({
        asset,
        decision: "REJECT",
        shareCount: 0,
        pricePerShare,
        totalCostUsdc: "0",
        reasoning: evaluation.reasoning,
        metricsAssessed: evaluation.metricsAssessed,
        txHashes: [],
      });
    }

    await waitMs(`rate limit before evaluating next asset (${label})`, 700);
  }

  // Phase 4: Report (planned)
  log("STEP", "=== Phase 4: Writing Report (before execution) ===");

  const plannedPurchases = decisions.filter((d) => d.decision === "PURCHASE");
  const report: AgentReport = {
    walletAddress: signer.address,
    timestamp: new Date().toISOString(),
    criteria,
    platformUrl,
    decisions,
    planned: true,
    summary: {
      totalAssets: assets.length,
      purchased: plannedPurchases.length,
      rejected: decisions.length - plannedPurchases.length,
      totalSharesAcquired: plannedSharesTotal,
      totalUsdcSpent: plannedUsdcTotal.toFixed(2),
    },
  };

  const reportPath = await writeReport(report);

  console.log("\n" + "=".repeat(70));
  console.log("  PLANNED DECISIONS (before execution)");
  console.log("=".repeat(70));
  console.log(`  Assets reviewed    : ${report.summary.totalAssets}`);
  console.log(`  Planned purchases   : ${report.summary.purchased}`);
  console.log(`  Rejected            : ${report.summary.rejected}`);
  console.log(`  Report written      : ${reportPath}`);
  console.log("=".repeat(70) + "\n");

  // Phase 5: Execute purchases
  log("STEP", "=== Phase 5: Executing Purchases ===");

  let totalSharesAcquired = 0;
  let totalUsdcSpentNum = 0;

  for (const d of decisions) {
    if (d.decision !== "PURCHASE" || d.shareCount === 0) continue;

    await waitMs("nonce propagation before next purchase", 2000);

    const label = assetDisplayName(d.asset);
    log("INFO", `Executing: ${label} (${d.shareCount} shares)`);

    const { txHashes, error } = await executePurchase(
      mcpClient,
      signer,
      d.asset,
      d.shareCount,
      caps
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

  report.planned = false;
  report.summary = {
    totalAssets: decisions.length,
    purchased: decisions.filter((x) => x.decision === "PURCHASE" && !x.error).length,
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

process.on("SIGINT", () => {
  log("INFO", "Interrupted - exiting.");
  process.exit(0);
});

function isDirectRun(): boolean {
  const entry = process.argv[1];
  if (!entry) return false;
  return import.meta.url === pathToFileURL(path.resolve(entry)).href;
}

if (isDirectRun()) {
  main().catch((err: unknown) => {
    console.error("Fatal:", err);
    process.exit(1);
  });
}
