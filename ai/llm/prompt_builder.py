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
<<<<<<< HEAD
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
=======
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
>>>>>>> ac7e83f81b2ee49e15de6fba6f33e739aef42954

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
<<<<<<< HEAD
    ) or "Not provided"
=======
    ) or "general skin-soothing and antioxidant properties"
>>>>>>> ac7e83f81b2ee49e15de6fba6f33e739aef42954

    uses = "; ".join(
        herb_information.get(
            "uses",
            []
        )
<<<<<<< HEAD
    ) or "Not provided"
=======
    ) or "traditional topical application and skincare"
>>>>>>> ac7e83f81b2ee49e15de6fba6f33e739aef42954

    precautions = "; ".join(
        herb_information.get(
            "precautions",
            []
        )
<<<<<<< HEAD
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
=======
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
>>>>>>> ac7e83f81b2ee49e15de6fba6f33e739aef42954

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