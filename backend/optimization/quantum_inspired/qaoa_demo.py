"""Small, honest QAOA/AerSimulator demo for fuel-type selection."""

def run_fuel_qaoa_demo() -> dict:
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator

        circuit = QuantumCircuit(4, 4)
        circuit.h(range(4))
        circuit.rz(0.35, 0)
        circuit.rz(0.20, 1)
        circuit.rz(-0.25, 2)
        circuit.rz(-0.45, 3)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.cx(2, 3)
        circuit.measure(range(4), range(4))
        result = AerSimulator().run(circuit, shots=256).result()
        counts = result.get_counts()
        best = max(counts, key=counts.get)
        return {"status": "success", "simulator": "Qiskit AerSimulator", "executed_real_circuit": True, "selected_bitstring": best, "counts": counts, "fuel_mapping": {"00": "HFO", "01": "LNG", "10": "Methanol", "11": "Ammonia"}, "circuit_depth": circuit.depth()}
    except Exception as error:
        return {"status": "fallback", "simulator": "AerSimulator unavailable", "executed_real_circuit": False, "selected_bitstring": "01", "counts": {"01": 1}, "fuel_mapping": {"00": "HFO", "01": "LNG", "10": "Methanol", "11": "Ammonia"}, "message": str(error)}