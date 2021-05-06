"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.run = void 0;
require("source-map-support/register");
const triplydb_1 = __importDefault(require("@triply/triplydb"));
async function run() {
    const app = triplydb_1.default.get({ token: process.env.TRIPLY_API_TOKEN });
    // get organization that contains the to query Dataset.
    const user = await app.getUser("SimoneColombo");
    // Get query information
    const query = await user.getQuery("Authors-Names-1");
    // for construct and describe queries
    const results = query.results().statements();
    // saving the results to file
    await results.toFile(`./results.ttl`);
}
exports.run = run;
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
