require("source-map-support/register");
import TriplyDB from "@triply/triplydb";
export async function run() {
  const app = TriplyDB.get({ token: process.env.TRIPLY_API_TOKEN });
  // get organization that contains the to query Dataset.
  const user = await app.getUser("SimoneColombo");
  // Get query information
  const query = await user.getQuery("Authors-Names-1")
  // for construct and describe queries
  const results = query.results().statements();
  // saving the results to file
  await results.toFile(`./results.ttl`);
}
run().catch((e) => {
  console.error(e);
  process.exit(1);
});
process.on("uncaughtException", function (err) {
  console.error("Uncaught exception", err);
  process.exit(1);
});
process.on("unhandledRejection", (reason, p) => {
  console.error("Unhandled Rejection at: Promise", p, "reason:", reason);
  process.exit(1);
});
