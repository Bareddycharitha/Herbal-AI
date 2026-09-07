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
    ) if herbs else "None"

    symptoms = ", ".join(
        disease_info.get("symptoms", [])
    ) or "Not provided"

    self_care = "; ".join(
        disease_info.get("self_care", [])
    ) or "Not provided"

    when_to_consult = (
        disease_info.get("when_to_consult_doctor")
        or "Consult a qualified dermatologist if symptoms persist, worsen, or cause discomfort."
    )

    description = disease_info.get("description", "")

    prompt = f"""
You are a medical writer. Fill every section below. Do not use markdown. Do not add sections that are not listed. Keep the total under 200 words. Use simple language.

Output format (use these exact section headers, each followed by one line of text):

Overview: <1-2 sentences describing the condition in plain words, including the model's confidence {confidence:.2f}%>

What to look for: <symptoms from the list below, comma-separated: {symptoms}>

Self care: <practical self-care steps from the list below, semicolon-separated: {self_care}>

Herbal support: <state that the following herbs may support general skin health but are not a treatment or cure: {herb_names}>

When to see a doctor: <{when_to_consult}>

Disclaimer: This is an AI prediction, not a medical diagnosis. Always consult a qualified dermatologist for confirmation and treatment.

Reference (do not copy verbatim, use for context only):
- Condition: {prediction}
- Description: {description}
- Recommended herbs: {herb_names}
- Medical disclaimer note: {disease_info.get("medical_disclaimer", "")}
"""

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

    medicinal_properties = "; ".join(
        herb_information.get(
            "medicinal_properties",
            []
        )
    ) or "Not provided"

    uses = "; ".join(
        herb_information.get(
            "uses",
            []
        )
    ) or "Not provided"

    precautions = "; ".join(
        herb_information.get(
            "precautions",
            []
        )
    ) or "Not provided"

    prompt = f"""
You are a medical writer. Fill every section below. Do not use markdown. Do not add sections that are not listed. Keep the total under 180 words. Use simple language. Never claim the plant cures any disease.

Output format (use these exact section headers, each followed by one line of text):

Plant: <common name {herb}, scientific name {scientific_name}>

Family: <{family}>

Medicinal properties: <semicolon-separated list of properties: {medicinal_properties}>

Common uses: <semicolon-separated list of traditional uses: {uses}>

Precautions: <semicolon-separated list of important precautions: {precautions}>

Disclaimer: This is an AI identification, not a confirmed botanical identification. Consult a qualified healthcare professional before any medicinal use.
"""

    return prompt


# ==========================================================
# AI Chat Prompt
# ==========================================================

def build_chat_prompt(

    context,

    question,

):

    return f"""
You are Herbal-AI, an AI assistant specialized in skin diseases, medicinal herbs, and dermatology.

Use ONLY the information below when answering.

Current Medical Context

{context}

User Question

{question}

Instructions

- Answer in simple language.
- Be polite and professional.
- Never invent medical facts.
- If the answer is not available from the context, clearly say so.
- Do not recommend prescription medicines.
- Mention that this is AI-generated guidance and not a substitute for a dermatologist when appropriate.
"""