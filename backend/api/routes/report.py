import io
import os
import sys

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from benchmarking.run_benchmarks import load_data
from optimization.optimizer import run_fleet_optimization

router = APIRouter(prefix="/api", tags=["Reports"])

def _pdf(text_lines):
    lines = "BT /F1 12 Tf 50 760 Td " + " ".join(f"({line.replace('(', '[').replace(')', ']')}) Tj 0 -18 Td" for line in text_lines) + " ET"
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>", "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>", "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>", f"<< /Length {len(lines.encode())} >>\nstream\n{lines}\nendstream"]
    body = "%PDF-1.4\n"
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(body.encode()))
        body += f"{index} 0 obj\n{obj}\nendobj\n"
    xref = len(body.encode())
    body += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n" + "".join(f"{offset:010d} 00000 n \n" for offset in offsets[1:])
    body += f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF"
    return body.encode()

@router.get("/case-study-report.pdf")
def case_study_report():
    vessels, routes = load_data()
    result = run_fleet_optimization(vessels[:6], routes, algorithm="QIGA", pop_size=8, generations=3, carbon_tax=100)
    best = result["result"].get("best_fitness", [0, 0, 0])
    pdf = _pdf(["GREEN FLEET OPTIMIZER - SIH 2026", "Quantum-inspired maritime fleet planning case study", f"Algorithm: {result['primary_algorithm']}", f"Cost: ${best[0]:,.0f}", f"WtW emissions: {best[1]:,.0f} tCO2e", f"Schedule delay: {best[2]:,.1f} hours", "Route, speed, fuel, shore power, compliance and Pareto optimization included."])
    return StreamingResponse(io.BytesIO(pdf), media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=green-fleet-case-study.pdf"})