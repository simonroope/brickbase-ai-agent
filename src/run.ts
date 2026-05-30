/**
 * Minimal entry point — logs immediately before loading the agent module.
 */
console.log(
  `[${new Date().toISOString()}] [>]  Process started — loading agent...`
);

const { main } = await import("./agent.js");

await main().catch((err: unknown) => {
  console.error("Fatal:", err);
  process.exit(1);
});
