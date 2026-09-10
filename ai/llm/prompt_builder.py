"""
Prompt templates for Herbal-AI.
These prompts are shared by both the AI Summary
and the AI Chatbot.
"""


# ==========================================================
# Skin Disease Summary Prompt
# ==========================================================

def build_summary_prompt(
    prediction,
    confidence,
    disease_info,
    herbs
):
    herb_names = ", ".join(
        herb["name"] for herb in herbs
    ) if herbs else "None specifically identified"

    symptoms = ", ".join(
        disease_info.get("symptoms", [])
    ) or "general skin redness, itching, or irritation"

    self_care = ", ".join(
        disease_info.get("self_care", [])
    ) or "gently cleanse the area, apply a mild fragrance-free moisturizer, and avoid scratching"

    when_to_see_doctor = disease_info.get("when_to_consult_doctor") or "if symptoms persist, worsen, or cause discomfort"

    description = disease_info.get("description") or f"a potential skin presentation identified as {prediction}"

    prompt = f"""Clinical Information:
- Detected Condition: {prediction}
- AI Confidence: {confidence:.1f}%
- Key Symptoms: {symptoms}
- Daily Self-Care: {self_care}
- Herbal Options: {herb_names}
- Doctor Consultation: {when_to_see_doctor}
- Summary: {description}

Directly write a 2-paragraph patient explanation in plain text (no markdown, no bullets, no asterisks, no reasoning or prompt commentary):
Paragraph 1: Explain the condition and the AI model's confidence in patient-friendly terms, advising a dermatologist visit to confirm.
Paragraph 2: Outline self-care steps and mention how suggested herbal remedies support wellness without replacing medical treatment."""

    return prompt


# ==========================================================
# Medicinal Herb Summary Prompt
# ==========================================================

def build_herb_summary_prompt(
    herb,
    herb_information,
):

    scientific_name = herb_information.get(
        "scientific_name",
        "Unknown"
    )

    family = herb_information.get(
        "family",
        "Unknown"
    )

    medicinal_properties = ", ".join(
        herb_information.get(
            "medicinal_properties",
            []
        )
    ) or "general skin-soothing and antioxidant properties"

    uses = ", ".join(
        herb_information.get(
            "uses",
            []
        )
    ) or "traditional topical application and skincare"

    precautions = ", ".join(
        herb_information.get(
            "precautions",
            []
        )
    ) or "perform a patch test before use and avoid on open wounds"

    prompt = f"""Botanical Information:
- Plant: {herb}
- Scientific Name: {scientific_name}
- Family: {family}
- Medicinal Properties: {medicinal_properties}
- Traditional Uses: {uses}
- Precautions: {precautions}

Directly write a 2-paragraph patient explanation in plain text (no markdown, no bullets, no asterisks, no reasoning or prompt commentary):
Paragraph 1: Introduce the plant with common and scientific names and summarize its main traditional uses and skin-supportive properties.
Paragraph 2: Explain precautions, note that AI identification is not a confirmed botanical diagnosis, and recommend consulting a healthcare provider before use."""

    return prompt


# ==========================================================
# AI Chat Prompt
# ==========================================================

def build_chat_prompt(

    context,

    question,

):

    return f"""A patient who has just received an AI-assisted skin analysis is asking a follow-up question. Answer helpfully using only the medical context provided. Write in clear, plain English. No markdown, no bullet points, no headings, no asterisks. Do not mention the prompt, your role, or any instructions.

If the answer is not available from the context, say so plainly. Never invent medical facts. Do not recommend prescription medicines. Remind the patient that this is AI-generated guidance and not a substitute for a dermatologist when appropriate.

Medical context:

{context}

Patient question: {question}
"""