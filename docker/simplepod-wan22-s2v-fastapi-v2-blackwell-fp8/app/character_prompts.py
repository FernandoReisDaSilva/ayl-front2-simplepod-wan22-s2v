NEGATIVE_PROMPT = (
    "fast head movement, head bobbing, jerky motion, excessive body swaying, exaggerated motion, overacting, "
    "distorted mouth, weak lip sync, blurry face, identity drift, singing performance, subtitles"
)

CHARACTER_PROMPTS = {
    "alex": (
        "stable square close-up talking head portrait of the same man, natural English speech articulation, "
        "stronger and more active accurate lip sync, clear mouth openings closures rounded vowels and labial consonants, "
        "slower more natural head motion, subtle controlled head turns and nods, gentle eye neck shoulder and torso micro movements, "
        "preserved identity, high quality face, natural conversational delivery in English"
    ),
    "sofi": (
        "stable square close-up talking head portrait of the same woman, natural Spanish speech articulation, "
        "stronger and more active accurate lip sync, clear mouth openings closures rounded vowels and labial consonants, "
        "slower more natural head motion, subtle controlled head turns and nods, gentle eye neck shoulder and torso micro movements, "
        "preserved identity, high quality face, natural conversational delivery in Spanish"
    ),
    "fernando": (
        "stable square close-up talking head portrait of the same man, natural Brazilian Portuguese speech articulation, "
        "stronger and more active accurate lip sync, clear mouth openings closures rounded vowels and labial consonants, "
        "slower more natural head motion, subtle controlled head turns and nods, gentle eye neck shoulder and torso micro movements, "
        "preserved identity, high quality face, natural conversational delivery in Brazilian Portuguese"
    ),
    "mae": (
        "stable square close-up talking head portrait of the same woman, natural French speech articulation, "
        "stronger and more active accurate lip sync, clear mouth openings closures rounded vowels and labial consonants, "
        "slower more natural head motion, subtle controlled head turns and nods, gentle eye neck shoulder and torso micro movements, "
        "preserved identity, high quality face, natural conversational delivery in French"
    ),
    "luca": (
        "stable square close-up talking head portrait of the same man, natural Italian speech articulation, "
        "stronger and more active accurate lip sync, clear mouth openings closures rounded vowels and labial consonants, "
        "slower more natural head motion, subtle controlled head turns and nods, gentle eye neck shoulder and torso micro movements, "
        "preserved identity, high quality face, natural conversational delivery in Italian"
    ),
}


def resolve_character_prompts(character_id: str) -> dict:
    normalized = str(character_id or "").strip().lower()
    if normalized not in CHARACTER_PROMPTS:
        expected = ", ".join(sorted(CHARACTER_PROMPTS))
        raise ValueError(f"Unsupported character_id for prompt resolution: {character_id!r}. Expected one of: {expected}")
    return {
        "character_id": normalized,
        "positive_prompt": CHARACTER_PROMPTS[normalized],
        "negative_prompt": NEGATIVE_PROMPT,
    }
