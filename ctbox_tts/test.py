from chatterbox_dialogue_tts import generate_dialogue_tts

result = generate_dialogue_tts(
    speaker1_name="Wang Xiaoming",
    speaker2_name="Wang Xiaohong",
    speaker1_text=[
        "We're finally here at the Grand Canyon, Xiaohong!",
        "I brought my camera to capture everything.",
    ],
    speaker2_text=[
        "Me too! The views are stunning already.",
        "Perfect! I'll record some videos on my phone.",
    ],
    output_dir="tts_outputs/example_001",
)

speaker1_tts = result["speaker1_tts"]  # left channel
speaker2_tts = result["speaker2_tts"]  # right channel
metadata = result["metadata"]