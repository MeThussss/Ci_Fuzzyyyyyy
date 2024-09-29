import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# สร้างตัวแปร input สำหรับระบบ Fuzzy
physical_violence = ctrl.Antecedent(np.arange(0, 11, 1), 'physical_violence')
property_damage = ctrl.Antecedent(np.arange(0, 101, 1), 'property_damage')
repeat_offense = ctrl.Antecedent(np.arange(0, 6, 1), 'repeat_offense')
intent = ctrl.Antecedent(np.arange(0, 11, 1), 'intent')
victim_impact = ctrl.Antecedent(np.arange(0, 11, 1), 'victim_impact')

# สร้างตัวแปร output สำหรับระบบ Fuzzy
punishment_severity = ctrl.Consequent(np.arange(0, 11, 1), 'punishment_severity')

# กำหนดชุด fuzzy สำหรับ input
physical_violence['low'] = fuzz.trimf(physical_violence.universe, [0, 0, 5])
physical_violence['medium'] = fuzz.trimf(physical_violence.universe, [0, 5, 10])
physical_violence['high'] = fuzz.trimf(physical_violence.universe, [5, 10, 10])

property_damage['low'] = fuzz.trimf(property_damage.universe, [0, 0, 50])
property_damage['medium'] = fuzz.trimf(property_damage.universe, [0, 50, 100])
property_damage['high'] = fuzz.trimf(property_damage.universe, [50, 100, 100])

repeat_offense['none'] = fuzz.trimf(repeat_offense.universe, [0, 0, 1])
repeat_offense['few'] = fuzz.trimf(repeat_offense.universe, [0, 2, 4])
repeat_offense['many'] = fuzz.trimf(repeat_offense.universe, [3, 5, 5])

intent['accidental'] = fuzz.trimf(intent.universe, [0, 0, 5])
intent['partial'] = fuzz.trimf(intent.universe, [0, 5, 10])
intent['full'] = fuzz.trimf(intent.universe, [5, 10, 10])

victim_impact['low'] = fuzz.trimf(victim_impact.universe, [0, 0, 5])
victim_impact['medium'] = fuzz.trimf(victim_impact.universe, [0, 5, 10])
victim_impact['high'] = fuzz.trimf(victim_impact.universe, [5, 10, 10])

# กำหนดชุด fuzzy สำหรับ output
punishment_severity['light'] = fuzz.trimf(punishment_severity.universe, [0, 0, 5])
punishment_severity['moderate'] = fuzz.trimf(punishment_severity.universe, [0, 5, 10])
punishment_severity['severe'] = fuzz.trimf(punishment_severity.universe, [5, 10, 10])

# สร้างกฎ (Fuzzy Rules)
rule1 = ctrl.Rule(physical_violence['low'] & property_damage['low'] & repeat_offense['none'] & intent['accidental'] & victim_impact['low'], punishment_severity['light'])
rule2 = ctrl.Rule(physical_violence['medium'] | property_damage['medium'] | repeat_offense['few'], punishment_severity['moderate'])
rule3 = ctrl.Rule(physical_violence['high'] | repeat_offense['many'] | victim_impact['high'] | intent['full'], punishment_severity['severe'])

# สร้างระบบควบคุม
punishment_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
punishment_decision = ctrl.ControlSystemSimulation(punishment_ctrl)

# รับค่า input จากผู้ใช้
print("กรุณาป้อนค่าระดับความรุนแรงต่างๆ ที่เกิดขึ้น:")
physical_violence_input = float(input("Physical Violence (0-10): "))
property_damage_input = float(input("Property Damage (0-100): "))
repeat_offense_input = float(input("Repeat Offense (0-5): "))
intent_input = float(input("Intent (0-10): "))
victim_impact_input = float(input("Victim Impact (0-10): "))

# กำหนดค่าตัวแปร input ที่ได้รับจากผู้ใช้
punishment_decision.input['physical_violence'] = physical_violence_input
punishment_decision.input['property_damage'] = property_damage_input
punishment_decision.input['repeat_offense'] = repeat_offense_input
punishment_decision.input['intent'] = intent_input
punishment_decision.input['victim_impact'] = victim_impact_input

# คำนวณผลลัพธ์
punishment_decision.compute()

# แสดงผลลัพธ์
print(f"ระดับการลงโทษ: {punishment_decision.output['punishment_severity']:.2f}/10")

# แสดงผล Fuzzy ของ input และ output เป็นกราฟ
physical_violence.view(sim=punishment_decision)
property_damage.view(sim=punishment_decision)
repeat_offense.view(sim=punishment_decision)
intent.view(sim=punishment_decision)
victim_impact.view(sim=punishment_decision)
punishment_severity.view(sim=punishment_decision)

plt.show()
