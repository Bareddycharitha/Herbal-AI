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
    )

    self_care = ", ".join(
        disease_info.get("self_care", [])
    )

    prompt = f"""
You are an experienced dermatologist.

Generate a professional medical summary.

Rules:

- Maximum 180 words
- Simple language
- No markdown
- Do not exaggerate certainty
- Mention this is an AI prediction, not a confirmed diagnosis.
- Recommend consulting a dermatologist when appropriate.
- If herbal recommendations are provided, explain that they may support skin health but are not a substitute for medical treatment.

Prediction:
{prediction}

Confidence:
{confidence:.2f}%

Description:
{disease_info.get("description", "")}

Symptoms:
{symptoms}

Self Care:
{self_care}

Recommended Herbs:
{herb_names}

When to Consult a Doctor:
{disease_info.get("when_to_consult_doctor", "")}

Medical Disclaimer:
{disease_info.get("medical_disclaimer", "")}
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

    medicinal_properties = ", ".join(
        herb_information.get(
            "medicinal_properties",
            []
        )
    )

    uses = ", ".join(
        herb_information.get(
            "uses",
            []
        )
    )

    precautions = ", ".join(
        herb_information.get(
            "precautions",
            []
        )
    )

    prompt = f"""
You are an expert botanist and Ayurvedic medicinal plant specialist.

Generate a professional medicinal plant summary.

Rules:

- Maximum 180 words
- Simple language
- No markdown
- Mention this is an AI identification and not a confirmed botanical identification.
- Explain the medicinal uses.
- Mention important precautions.
- Never claim the plant cures diseases.
- Recommend consulting a healthcare professional before medicinal use.

Plant:
{herb}

Scientific Name:
{scientific_name}

Family:
{family}

Medicinal Properties:
{medicinal_properties}

Uses:
{uses}

Precautions:
{precautions}
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