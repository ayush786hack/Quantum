import React, { useEffect, useState } from "react";
import Dashboard from "./components/Dashboard/Dashboard";
import { getBenchmark, getCaseStudy, optimizeFleet } from "./services/api";

const fallback = {
	summary: { optimized_cost_usd: 3920000, optimized_emissions_tco2e: 14200, cost_reduction_pct: 19.2, emission_reduction_pct: 63.1 },
	fleet_size: 12, scenario_name: "SIH 2026 Green Corridor 2030 Demonstration"
};

export default function App() {
	const [caseStudy, setCaseStudy] = useState(fallback);
	const [result, setResult] = useState(null);
	const [benchmark, setBenchmark] = useState(null);
	const [loading, setLoading] = useState(false);
	const [notice, setNotice] = useState("Demo data ready");

	useEffect(() => {
		Promise.allSettled([getCaseStudy(), getBenchmark()]).then(([scenario, bench]) => {
			if (scenario.status === "fulfilled") setCaseStudy(scenario.value);
			if (bench.status === "fulfilled") setBenchmark(bench.value);
			if (scenario.status === "rejected") setNotice("Preloaded case study · API offline");
		});
	}, []);

	async function handleOptimize(scenario) {
		setLoading(true); setNotice("Evolving fleet plans...");
		try {
			const data = await optimizeFleet(scenario);
			setResult(data); setNotice(`${data.primary_algorithm} completed in ${data.total_execution_time_seconds}s`);
		} catch {
			setNotice("API unavailable · showing the preloaded case study");
		} finally { setLoading(false); }
	}

	return <Dashboard caseStudy={caseStudy} result={result} benchmark={benchmark} loading={loading} notice={notice} onOptimize={handleOptimize} />;
}
