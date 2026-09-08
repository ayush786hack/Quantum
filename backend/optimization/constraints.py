def validate_fleet_constraints(fleet_plan: list, max_allowed_cii_grade: str = "C") -> dict:
    """
    Validates hard and soft constraints for a proposed fleet allocation.
    Returns boolean status and list of violation messages.
    """
    grade_order = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
    max_level = grade_order.get(max_allowed_cii_grade, 3)
    
    violations = []
    
    for idx, item in enumerate(fleet_plan):
        v = item['vessel']
        r = item['route']
        speed = item['speed_knots']
        fuel = item['fuel_type']
        sp_used = item['shore_power_used']
        
        # Speed bounds check
        min_s = v.get('base_speed_knots', 16.0) * 0.65
        max_s = v.get('base_speed_knots', 16.0) * 1.25
        if not (min_s <= speed <= max_s):
            violations.append(f"Vessel {v['vessel_id']}: Speed {speed:.1f} kts out of bounds [{min_s:.1f}, {max_s:.1f}].")
            
        # Fuel compatibility check
        if fuel not in v.get('supported_fuels', ['HFO']):
            violations.append(f"Vessel {v['vessel_id']}: Fuel '{fuel}' not supported by engine.")
            
        # Shore power port availability check
        if sp_used:
            port_sp = r.get('shore_power_origin', False) and r.get('shore_power_dest', False)
            if not port_sp:
                violations.append(f"Vessel {v['vessel_id']} on route {r['id']}: Shore power requested but unavailable at berth.")
                
    is_valid = (len(violations) == 0)
    return {
        "is_valid": is_valid,
        "violation_count": len(violations),
        "violations": violations
    }
