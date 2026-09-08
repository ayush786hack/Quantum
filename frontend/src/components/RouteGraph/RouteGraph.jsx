import React from "react";

export default function RouteGraph({ data, loading, trackData, tracking, onToggleTracking }) {
  const options = data?.options || [];
  const width = 760;
  const height = 310;
  const positions = [92, 380, 668];

  return (
    <section className="route-graph-panel">
      <div className="chart-note">
        <span className="legend-dot cyan" /> LIVE VOYAGE MAP / FUEL-AWARE CORRIDOR CHOICE
        <small>{loading ? "Recalculating with prediction engine..." : data?.fuel_type ? `Fuel: ${data.fuel_type}` : "Select a Pareto plan"}</small>
      </div>
      {options.length ? (
        <>
          <div className="map-toolbar"><span><i className="map-dot origin" /> ORIGIN</span><span><i className="map-dot vessel" /> VESSEL PLAN</span><span><i className="map-dot destination" /> DESTINATION</span><span className="map-status">{loading ? "UPDATING" : tracking ? "LIVE AIS" : "STANDBY"}</span><button className="track-button" onClick={onToggleTracking}>{tracking ? "STOP TRACKING" : "START LIVE TRACKING"}</button></div>
          <svg className="route-graph real-map" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Fuel-aware route corridor comparison">
            <defs>
              <pattern id="nautical-grid" width="38" height="38" patternUnits="userSpaceOnUse"><path d="M 38 0 L 0 0 0 38" fill="none" stroke="#29454b" strokeWidth="1" opacity=".45" /></pattern>
              <filter id="route-glow"><feGaussianBlur stdDeviation="3" result="blur" /><feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
            </defs>
            <rect width={width} height={height} fill="#0b2229" />
            <rect width={width} height={height} fill="url(#nautical-grid)" />
            <path d="M0 0H160L190 42L155 76L178 118L132 155L155 205L108 244L0 260Z" className="map-land" />
            <path d="M760 0H625L600 45L632 88L588 126L620 170L574 210L608 260L760 275Z" className="map-land" />
            <path d="M0 265C190 238 310 286 455 258S650 240 760 268" className="map-current" />
            {options.map((option, index) => {
              const y = 65 + index * 76;
              const recommended = option.id === data.recommended_route_id;
              return (
                <g key={option.id}>
                  <path d={`M ${positions[0]} ${y} C 230 ${y + (index - 1) * 32}, 300 ${y - (index - 1) * 34}, ${positions[1]} ${y} S 540 ${y + (index - 1) * 34}, ${positions[2]} ${y}`} className={recommended ? "route-line recommended" : "route-line"} filter={recommended ? "url(#route-glow)" : undefined} />
                  <circle cx={positions[0]} cy={y} r="7" className="route-node origin-node" />
                  <circle cx={positions[2]} cy={y} r="7" className="route-node destination-node" />
                  {recommended && <path d={`M ${positions[1] - 12} ${y - 8} L ${positions[1] + 12} ${y} L ${positions[1] - 12} ${y + 8} Z`} className="vessel-marker" />}
                  <text x="205" y={y - 17} className="route-label">{option.label}</text>
                  <text x="205" y={y + 24} className="route-metric">{option.distance_nmi.toLocaleString()} nmi · {option.fuel_consumption_tonnes} t fuel</text>
                  {recommended && <text x="548" y={y - 17} className="route-badge">LOWEST FUEL</text>}
                </g>
              );
            })}
            <text x="44" y="296" className="route-port">{data.origin}</text>
            <text x="606" y="296" className="route-port">{data.destination}</text>
            {trackData && <g className="live-position"><circle cx={positions[0] + ((positions[2] - positions[0]) * trackData.progress_pct) / 100} cy={65 + ((data.options.findIndex((item) => item.id === data.recommended_route_id)) * 76)} r="9" /><text x="18" y="24" className="live-position-label">LIVE {trackData.position.label}</text></g>}
          </svg>
          <div className="route-summary">{data.recommendation_reason} <strong>{data.recommended_route_id}</strong></div>
          {trackData && <div className="tracking-metrics"><div><span>CONSUMED</span><strong>{trackData.fuel_consumed_tonnes} t</strong></div><div><span>REMAINING</span><strong>{trackData.fuel_remaining_tonnes} t</strong></div><div><span>BURN RATE</span><strong>{trackData.fuel_rate_tonnes_per_hour} t/h</strong></div><div><span>ETA</span><strong>{trackData.eta_hours} h</strong></div><small>{trackData.source} · {trackData.distance_completed_nmi} / {trackData.distance_completed_nmi + trackData.distance_remaining_nmi} nmi complete</small></div>}
        </>
      ) : <div className="route-empty">Run the optimizer to compare direct, sheltered, and green corridors.</div>}
    </section>
  );
}
