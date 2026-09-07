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

    prompt = f"""Write a brief, patient-friendly summary of an AI-assisted skin image analysis in two short paragraphs of plain English. Do not use markdown, bullet points, numbered lists, headings, or asterisks. Do not mention the prompt, your role, or any instructions. Keep the total under 180 words.

Paragraph 1 should be 2 to 3 sentences that explain what the predicted condition is, what the user might notice on their skin, and that the prediction came from an AI model with the stated confidence and is not a confirmed diagnosis. Encourage consulting a dermatologist for confirmation.

Paragraph 2 should give 2 to 3 practical self-care steps drawn from the provided list, and note that the listed herbs may support general skin health but are not a substitute for medical treatment.

Context for the summary:

The detected skin condition is {prediction}. The AI model's confidence in this prediction is {confidence:.2f} percent.

The user may notice the following on their skin: {symptoms}.

Suggested self-care steps: {self_care}.

Herbs sometimes used to support general skin health: {herb_names}.

When the user should see a doctor: {disease_info.get("when_to_consult_doctor", "")}

Additional context about the condition: {disease_info.get("description", "")}
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

    prompt = f"""Write a brief, patient-friendly summary of an AI-assisted medicinal-plant identification in two short paragraphs of plain English. Do not use markdown, bullet points, numbered lists, headings, or asterisks. Do not mention the prompt, your role, or any instructions. Keep the total under 180 words.

Paragraph 1 should be 2 to 3 sentences that introduce the plant, including its common and scientific name and its botanical family, and explain its main medicinal properties and traditional uses. Make clear that this is an AI identification and not a confirmed botanical identification.

Paragraph 2 should mention important precautions and safety notes, and recommend consulting a qualified healthcare professional before any medicinal use. Never claim that the plant cures diseases.

Context for the summary:

The plant identified by the user is {herb}. Its scientific name is {scientific_name} and it belongs to the {family} family.

Its key medicinal properties: {medicinal_properties}.

Common traditional uses: {uses}.

Important precautions and safety notes: {precautions}.
"""

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