"""
SmartGrid PredictAI - Transformer Engineering Reference Data
================================================================
Hand-written reference content (not fabricated telemetry, not model
output) consumed by dashboard/chatbot.py for two rule-based intents:
free-text fault-symptom diagnosis and glossary/"what is X" lookups.
Kept as plain dicts, same spirit as the model-rationale prose already in
README.md - real engineering knowledge, deterministic keyword matching,
no LLM involved.

SYMPTOM_CAUSES and GLOSSARY use deliberately non-overlapping vocabularies
so dashboard/chatbot.py's substring dispatch (glossary checked first,
longest key first) can't ambiguously match both.
"""

SYMPTOM_CAUSES = {
    "humming": {
        "causes": [
            "Loose core laminations or clamping",
            "Magnetostriction from harmonic loading",
            "Overexcitation (supply voltage above nameplate rating)",
            "Loose mounting hardware coupling vibration into the structure",
        ],
        "inspection_steps": [
            "Check supply voltage against the nameplate rating",
            "Inspect and re-torque core clamping bolts",
            "Run a vibration/acoustic scan and compare to baseline",
            "Check for loose or missing mounting hardware",
        ],
    },
    "vibration": {
        "causes": [
            "Loose core or winding clamping",
            "Cooling fan imbalance or bearing wear",
            "Loose mounting/foundation bolts",
            "Winding deformation from a prior through-fault current",
        ],
        "inspection_steps": [
            "Inspect cooling fan blades and bearings",
            "Torque-check mounting and core clamping bolts",
            "Run a winding frequency response analysis (FRA) if a through-fault is suspected",
            "Compare the vibration signature to its commissioning baseline",
        ],
    },
    "overheating": {
        "causes": [
            "Sustained overloading beyond the nameplate kVA rating",
            "Blocked or degraded cooling (radiators/fans)",
            "Low or degraded oil level/quality",
            "Poor ventilation at the installation site",
        ],
        "inspection_steps": [
            "Check load against the nameplate kVA rating",
            "Inspect radiators and cooling fans for blockage or failure",
            "Sample oil for dissolved gas analysis (DGA)",
            "Verify oil level and top up or replace if needed",
        ],
    },
    "oil leak": {
        "causes": [
            "Degraded or damaged gaskets/seals",
            "Corroded tank or radiator",
            "Overpressure from internal fault gas generation",
            "Loose valve or fitting",
        ],
        "inspection_steps": [
            "Visually trace the leak source (gasket, weld seam, valve, radiator)",
            "Check oil level and top up to compensate for loss",
            "Tighten or replace the affected gasket/fitting",
            "If overpressure is suspected, sample oil for DGA before further action",
        ],
    },
    "smoke": {
        "causes": [
            "Severe internal winding or insulation fault",
            "Overheated bushing or terminal connection",
            "Oil breakdown from sustained overheating",
            "External arcing at a terminal",
        ],
        "inspection_steps": [
            "De-energize and isolate the transformer immediately - this is a safety-critical symptom",
            "Do not re-energize until it has been inspected",
            "Sample oil for DGA once it's safe to do so",
            "Inspect bushings and terminal connections for arcing damage",
        ],
    },
    "burning smell": {
        "causes": [
            "Overheated bushing or terminal connection",
            "Oil breakdown from sustained overheating",
            "Insulation (paper/varnish) charring from a hot spot",
        ],
        "inspection_steps": [
            "De-energize and isolate the transformer if the smell is strong or worsening",
            "Thermally scan all terminal connections",
            "Sample oil for DGA once safe to do so",
        ],
    },
    "popping sound": {
        "causes": [
            "Partial discharge (internal arcing in the insulation)",
            "A loose electrical connection arcing intermittently",
            "Moisture ingress causing internal flashover",
        ],
        "inspection_steps": [
            "Run a partial discharge test",
            "Thermally scan all terminal connections",
            "Sample oil for moisture content and DGA",
        ],
    },
    "frequent tripping": {
        "causes": [
            "Internal fault (winding or insulation failure)",
            "An external fault on the connected feeder",
            "Protection relay miscalibration",
            "Sustained overload triggering thermal protection",
        ],
        "inspection_steps": [
            "Review protection relay event/fault records for fault type and location",
            "Sample oil for DGA to rule out an internal fault",
            "Verify protection relay settings against the transformer's rating",
            "Check load history for sustained overload",
        ],
    },
}

GLOSSARY = {
    "dissolved gas analysis": {
        "meaning": "A laboratory test that measures gases dissolved in a transformer's "
                   "insulating oil (e.g. hydrogen, methane, acetylene) to detect early "
                   "signs of internal faults.",
        "causes": ["Not a fault itself - a diagnostic test triggered by suspected "
                    "overheating, arcing, or partial discharge"],
        "consequences": "Elevated gas levels indicate an active internal fault (thermal "
                         "or electrical) that will worsen if left untreated.",
        "recommended_actions": [
            "Compare results against IEEE C57.104 / IEC 60599 gas-ratio guidelines",
            "Repeat sampling to establish a gassing trend rate",
            "Schedule internal inspection if gas levels or trend rate are high",
        ],
    },
    "insulation failure": {
        "meaning": "Breakdown of the solid (paper/pressboard) or liquid (oil) insulation "
                   "separating energized windings from each other and from the tank, "
                   "allowing current to flow where it shouldn't.",
        "causes": [
            "Moisture ingress into the oil or paper insulation",
            "Long-term thermal aging (cellulose insulation degrades faster at sustained high temperature)",
            "Mechanical damage from through-fault currents",
            "Contamination or oil oxidation",
        ],
        "consequences": "Progresses to partial discharge, then flashover/short circuit - "
                         "typically a catastrophic, non-repairable failure mode.",
        "recommended_actions": [
            "Test insulation resistance and power factor (tan delta)",
            "Sample oil for moisture content and DGA",
            "Review loading history for prior through-fault events",
        ],
    },
    "partial discharge": {
        "meaning": "Small, localized electrical discharges that partially bridge the "
                   "insulation between conductors without fully short-circuiting it - an "
                   "early warning sign of insulation degradation.",
        "causes": [
            "Voids or gas bubbles in solid insulation",
            "Moisture or contamination in the oil",
            "Sharp edges or points creating a high local electric field",
        ],
        "consequences": "Left unaddressed, it erodes insulation over time and can "
                         "progress to full insulation failure.",
        "recommended_actions": [
            "Run a partial discharge (PD) test",
            "Sample oil for DGA (look for a hydrogen/methane signature)",
            "Inspect for moisture ingress points",
        ],
    },
    "bushing failure": {
        "meaning": "Failure of the insulated terminal (bushing) that carries a conductor "
                   "through the grounded tank wall - a common transformer failure point.",
        "causes": [
            "Moisture ingress into the bushing insulation",
            "Partial discharge/tracking on the bushing surface",
            "Manufacturing defect or age-related insulation degradation",
            "A loose or overheated bushing connection",
        ],
        "consequences": "Can progress to a violent bushing explosion or fire - a "
                         "well-known catastrophic transformer failure mode.",
        "recommended_actions": [
            "Test bushing power factor (tan delta) and capacitance",
            "Thermally scan the bushing terminal connection",
            "Sample oil for DGA if the bushing is oil-filled",
        ],
    },
    "high oil temperature": {
        "meaning": "Oil temperature running above its normal operating range for the "
                   "transformer's load and ambient conditions.",
        "causes": [
            "Sustained overloading beyond the nameplate rating",
            "Blocked or failed cooling (radiators/fans)",
            "Degraded oil quality reducing heat dissipation",
            "High ambient temperature or poor site ventilation",
        ],
        "consequences": "Accelerates insulation aging - roughly, every 6-8degC of "
                         "sustained overheating can halve the remaining insulation life.",
        "recommended_actions": [
            "Check load against the nameplate kVA rating",
            "Inspect and clean/repair cooling equipment",
            "Sample oil for DGA if the temperature rise is unexplained by load alone",
        ],
    },
    # This project's five raw model features, explained in engineering terms
    # so a user can ask e.g. "what does oil_quality_index mean" after seeing
    # it in a top_reasons string and get a direct answer.
    "temperature_rise_c": {
        "meaning": "This project's model feature for how far a transformer's operating "
                   "temperature has risen above ambient, in degC - a real thermal "
                   "measurement, not a heuristic.",
        "causes": ["Load, cooling condition, and oil quality all drive it up"],
        "consequences": "One of the five features driving this project's risk score - "
                         "sustained high values accelerate insulation aging.",
        "recommended_actions": ["See 'high oil temperature' for inspection steps"],
    },
    "oil_quality_index": {
        "meaning": "This project's 0-1 composite score for insulating oil condition "
                    "(higher is better), standing in for measurements like dielectric "
                    "breakdown voltage, moisture content, and acidity in a real oil "
                    "test report.",
        "causes": ["Degrades over time from thermal aging, moisture ingress, and oxidation"],
        "consequences": "Poor oil quality accelerates insulation aging and is one of "
                         "the five features driving this project's risk score.",
        "recommended_actions": [
            "In a real deployment: sample oil for dielectric breakdown voltage, moisture (ppm), and acidity",
            "Replace or reclaim (filter/dry) oil if quality has degraded significantly",
        ],
    },
    "load_factor": {
        "meaning": "This project's 0-1 measure of how heavily loaded a transformer is "
                    "relative to its rated capacity - sustained values near 1.0 mean "
                    "the unit is running near or above its nameplate rating.",
        "causes": ["Demand growth on the feeder, or undersized capacity for the connected load"],
        "consequences": "Sustained high load factor accelerates thermal aging and is "
                         "one of the five features driving this project's risk score.",
        "recommended_actions": [
            "Check actual load against the nameplate kVA rating",
            "Consider load redistribution or a capacity upgrade if consistently overloaded",
        ],
    },
    "maintenance_score": {
        "meaning": "This project's 0-1 composite score standing in for maintenance "
                    "history quality (higher is better) - inspection frequency, "
                    "servicing thoroughness, and issue resolution.",
        "causes": ["Deferred or inconsistent maintenance schedules"],
        "consequences": "Low maintenance_score is one of the five features driving "
                         "this project's risk score - it also correlates with worse "
                         "oil quality in the underlying data.",
        "recommended_actions": ["Bring the unit's inspection and servicing schedule up to date"],
    },
    "age_years": {
        "meaning": "This project's model feature for transformer age in years since "
                    "installation.",
        "causes": ["Simply time in service"],
        "consequences": "Older units are more likely to fail and have less remaining "
                         "useful life - one of the five features driving this "
                         "project's risk score.",
        "recommended_actions": ["Prioritize condition-based inspection as age increases, rather than age alone"],
    },
}
GLOSSARY["dga"] = GLOSSARY["dissolved gas analysis"]

# Ordered recommended-action lists per risk tier, deliberately aligned with
# models/failure_prediction.py's TIER_TO_MAINTENANCE_INTERVAL_DAYS (7/30/90/180
# days for critical/elevated/moderate/low) so the wording matches the actual
# next_maintenance_date the model computes.
MAINTENANCE_ACTIONS = {
    "critical": [
        "Inspect within 24 hours",
        "Perform a thermal scan",
        "Sample oil for dissolved gas analysis (DGA)",
        "Test insulation resistance",
        "Schedule emergency maintenance or consider a load transfer",
    ],
    "elevated": [
        "Inspect within 7 days",
        "Perform a thermal scan",
        "Check oil level and quality",
        "Inspect bushings and connections",
        "Schedule maintenance within 30 days",
    ],
    "moderate": [
        "Check oil level",
        "Review the load trend at the next routine visit",
        "Schedule maintenance within 90 days",
    ],
    "low": [
        "No immediate action required",
        "Continue the routine inspection schedule",
        "Recheck at the next scheduled maintenance (within 180 days)",
    ],
}
