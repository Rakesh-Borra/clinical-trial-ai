from agents.protocol_analysis_agent import analyze_protocol


sample_protocol = """
PROTOCOL TITLE: Type 2 Diabetes Treatment Study
PROTOCOL ID: T2D-001

INCLUSION CRITERIA:
1. Participants must be between 18 and 65 years old.
2. Participants must have a confirmed diagnosis of Type 2 Diabetes.

EXCLUSION CRITERIA:
1. Participants with severe kidney disease are excluded.

STUDY PROCEDURES:
1. Blood glucose testing.
2. Blood pressure measurement.

VISIT SCHEDULE:
Screening Visit: Day 0.
Follow-up Visit: Day 30.

SAFETY INFORMATION:
Participants will be monitored for adverse events.
"""


analysis = analyze_protocol(sample_protocol)

print("\n=== CLINICAL TRIAL PROTOCOL ANALYSIS ===\n")
print(analysis.model_dump_json(indent=2))