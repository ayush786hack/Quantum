import express from "express";
import { createProxyMiddleware } from "http-proxy-middleware";

const app = express();
const port = process.env.PORT || 3001;
const pythonService = process.env.PYTHON_SERVICE_URL || "http://127.0.0.1:8000";

app.get("/health", (_req, res) => res.json({ service: "green-fleet-gateway", status: "healthy", pythonService }));
app.use("/api", createProxyMiddleware({ target: pythonService, changeOrigin: true }));
app.listen(port, () => console.log(`Green Fleet gateway listening on ${port}`));