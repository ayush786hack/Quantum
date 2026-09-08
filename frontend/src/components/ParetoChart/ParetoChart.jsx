import React from "react";
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ZAxis } from "recharts";

export default function ParetoChart({ data, onSelectPoint }) {
  const formattedData = (data || []).map((item, idx) => ({
    ...item,
    cost_usd: item.cost_usd || item.fitness?.[0] || 4000000,
    emissions_tco2e: item.emissions_tco2e || item.fitness?.[1] || 15000,
    pointIndex: idx
  }));

  return (
    <div className="chart">
      <ResponsiveContainer width="100%" height={320}>
        <ScatterChart margin={{ top: 20, right: 30, bottom: 20, left: 20 }}>
          <CartesianGrid stroke="#26343a" strokeDasharray="3 5" />
          <XAxis 
            type="number" 
            dataKey="cost_usd" 
            name="Voyage Cost" 
            tickFormatter={(v) => `$${(v / 1e6).toFixed(2)}M`} 
            stroke="#718087" 
            domain={['dataMin - 100000', 'dataMax + 100000']}
          />
          <YAxis 
            type="number" 
            dataKey="emissions_tco2e" 
            name="WtW Emissions" 
            tickFormatter={(v) => `${Math.round(v / 1000)}k t`} 
            stroke="#718087"
            domain={['dataMin - 1000', 'dataMax + 1000']}
          />
          <ZAxis range={[90, 90]} />
          <Tooltip 
            cursor={{ strokeDasharray: "3 3" }} 
            formatter={(value, name) => [
              name === "Voyage Cost" ? `$${Number(value).toLocaleString()}` : `${Number(value).toLocaleString()} tCO2e`, 
              name
            ]} 
            contentStyle={{ background: "#122126", border: "1px solid #38535b", color: "#e8f1ed", borderRadius: "6px" }} 
          />
          <Scatter 
            data={formattedData} 
            fill="#c9f05b" 
            onClick={(e) => {
              if (e && e.pointIndex !== undefined && onSelectPoint) {
                onSelectPoint(e.pointIndex);
              }
            }}
            style={{ cursor: "pointer" }}
          />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
