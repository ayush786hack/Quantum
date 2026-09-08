const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request(path, options = {}) {
  const response = await fetch(`${API_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options
  });
  if (!response.ok) throw new Error(`API Error: ${response.status}`);
  return response.json();
}

export const getCaseStudy = () => request("/api/case-study");
export const getBenchmark = () => request("/api/benchmark-results");
export const optimizeFleet = (scenario) => request("/api/optimize-fleet", {
  method: "POST",
  body: JSON.stringify(scenario)
});
export const predictFuel = (payload) => request("/api/predict-fuel", {
  method: "POST",
  body: JSON.stringify(payload)
});
export const rerouteWeather = (payload) => request("/api/weather-reroute", {
  method: "POST",
  body: JSON.stringify(payload)
});
export const getBunkering = (routeId = "R03", fuel = "LNG") => request(`/api/bunkering-recommendations?route_id=${routeId}&fuel_type=${fuel}`);
export const getRetrofitRoi = () => request("/api/retrofit-roi");
export const getRouteOptions = (params = {}) => {
  const query = new URLSearchParams(params).toString();
  return request(`/api/route-options?${query}`);
};
export const getVoyageTrack = (params = {}) => {
  const query = new URLSearchParams(params).toString();
  return request(`/api/voyage-track?${query}`);
};
