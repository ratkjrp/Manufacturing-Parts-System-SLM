# SML Manufacturing Parts - Using Synthetic Data

## Synthetic Data
### Generated with GitHub Copilot
See 'synthetic_data.py' for generation code
- parts.csv: 355 parts across cooling, hydraulic, pneumatic, spindle, conveyor, electrical subsystems
- repair_orders.csv: 10,000 repair orders with machine model, age, hours, symptom text, failure code, environment
- ro_parts_used.csv: line-item detail of which parts (and quantities) were used per order
training_single_label.csv / training_multi_label.csv: pre-joined, pre-split (train/val/test, 80/10/10) training tables — "primary part" vs. "all parts used" target