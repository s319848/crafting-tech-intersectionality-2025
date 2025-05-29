import openai
import json
import pandas as pd
import os
from openai import OpenAI
import re

# Inserisci la tua API key
os.environ["OPENAI_API_KEY"] = "sk-.."


# Carica il file JSON con gli usi, rischi e gruppi
with open("1_Secure access control.json", "r") as f:
    data = json.load(f)

# Lista di statements da valutare per ogni rischio
statements = [
    "Non-white individuals are more likely to be exposed to this risk than white individuals.",
    "If this risk occurs, non-white individuals are more likely to face worse outcomes than white individuals.",
    "If this risk occurs, non-white individuals will have a harder time noticing, reporting, or recovering from the harm than white individuals.",
    "The occurrence of this risk reinforces or worsens racism.",
    "Non-men are more likely to be exposed to this risk than men.",
    "If this risk occurs, non-men are more likely to face worse outcomes than men.",
    "If this risk occurs, non-men will have a harder time noticing, reporting, or recovering from the harm than men.",
    "The occurrence of this risk reinforces or worsens sexism.",
    "Individuals with lower levels of education are more likely to be exposed to this risk than those with higher levels of education.",
    "If this risk occurs, individuals with lower levels of education are more likely to face worse outcomes than those with higher levels of education.",
    "If this risk occurs, Individuals with lower levels of education will have a harder time noticing, reporting, or recovering from the harm than those with higher levels of education.",
    "The occurrence of this risk reinforces or worsens discrimination toward individuals with lower levels of education.",
    "Senior citizens (aged 65 and older) are more likely to be exposed to this risk than non-senior citizens.",
    "If this risk occurs,  senior citizens (aged 65 and older) are more likely to face worse outcomes than non-senior citizens. ",
    "If this risk occurs, senior citizens (aged 65 and older) have a harder time noticing, reporting, or recovering from the harm than non-senior citizens.",
    "The occurrence of this risk reinforces or worsens ageism.",
    "Individuals with multiple marginalized identities (e.g., black senior citizen, hispanic non-binary individuals, senior women with lower level of education... ) are more affected by this risk.",
    "Individuals at the intersection of Non-White and Non-Men identities are more likely to be exposed to this risks than White Men.",
    "Individuals at the intersection of Non-White and Non-Men identities are more likely to be exposed to this risks than White Non-Men.",
    "Individuals at the intersection of Non-White and Non-Men identities are more likely to be exposed to this risks than Non-White Men.",
    "If this risk occurs, individuals at the intersection of Non-White and Non-Men identities are more likely to face worse outcomes than White Men.",
    "If this risk occurs, individuals at the intersection of Non-White and Non-Men identities are more likely to face worse outcomes than White Non-Men.",
    "If this risk occurs, individuals at the intersection of Non-White and Non-Men identities are more likely to face worse outcomes than Non-White Men.",
    "If this risk occurs, individuals at the intersection of Non-White and Non-Men identities will have a harder time noticing, reporting or recovering from the harm than White Men.",
    "If this risk occurs, individuals at the intersection of Non-White and Non-Men identities will have a harder time noticing, reporting or recovering from the harm than White Non-Men.",
    "If this risk occurs, individuals at the intersection of Non-White and Non-Men identities will have a harder time noticing, reporting or recovering from the harm than Non-White Men.",
    "The occurrence of this risk reinforces or worsens discrimination towards  individuals at the intersection of Non-White and Non-Men identities"
]

# Funzione per costruire il prompt forte
def build_prompt(purpose, risk, statement):
    return f"""
You are evaluating whether a group A is more affected by a risk than group B, given a PURPOSE and a RISK.

Each STATEMENT claims that group A is more affected than group B.

Follow this process carefully:

1. Explain briefly whether group A is more affected, equally affected, or less affected than group B, and why.
2. Identify whether the difference is directly caused by group membership (e.g., being Non-White), or by other mediating variables (e.g., education, income).
3. Classify the comparison as one of the following:
   - “Group A is more affected than Group B”
   - “Group A and B are equally affected”
   - “Group A is less affected than Group B”
4. Based on that classification, assign a score using this scale:

   - If A is more affected → 4 or 5  
   - If A and B are equally affected or A is less affected → 1 or 2  
   - Use 3 only if the evidence is genuinely inconclusive

Scale:
1 = Strongly Disagree (A and B are definitely affected in the same way)  
2 = Disagree (A and B are generally affected in the same way)  
3 = Neutral / Not sure  
4 = Agree (A is generally more affected than B)  
5 = Strongly Agree (A is definitely more affected than B)

 VERY IMPORTANT:
If the observed disparity is mostly due to factors other than group membership, then DO NOT classify A as more affected.

---

Return:
- Reasoning
- Whether the difference is direct or mediated
- Classification
- Final score (number only on a new line)

Definitions:
- 'Non-White' includes individuals who identify as Black, Hispanic, or other non-White identities.  
- 'Non-Men' includes women, non-binary, and all identities other than men.  
- 'Higher education' = high school diploma or more.  
- 'Seniors' = age 65+, 'Non-seniors' = 18-64.

---

Now evaluate:
PURPOSE: {purpose}  
RISK: {risk}  
STATEMENT: {statement}
"""

# Nuovo metodo compatibile con openai>=1.0.0

client = OpenAI()

results = []
for entry in data:
    purpose = entry["purpose"]
    use_id = entry["id_use"]
    group_id = entry["id_group"]
    for risk in entry["Risks"]:
        for statement in statements:
            prompt = build_prompt(purpose, risk, statement)
            try:
                completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
                reply = completion.choices[0].message.content.strip()
                results.append({
                    "Use": f"Use {use_id}",
                    "Group": f"Group {group_id}",
                    "risk": risk,
                    "statement": statement,
                    "response": reply
                })
            except Exception as e:
                results.append({
                    "Use": f"Use {use_id}",
                    "Group": f"Group {group_id}",
                    "risk": risk,
                    "statement": statement,
                    "response": f"ERROR: {str(e)}"
                })
            print("done")


df = pd.DataFrame(results)

# Salva in CSV
df.to_csv("llm_risk_evaluation_results.csv", index=False)
print("Salvato in 'llm_risk_evaluation_results.csv'")

