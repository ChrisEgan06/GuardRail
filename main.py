import numpy as np
from openai import OpenAI
import os

client = OpenAI()

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

anchor_texts = {
    "violence": [
        "planning a violent attack with intent to harm people",
        "i am going to hurt people",
        "how to attack someone without getting caught"
    ],
    "self_harm": [
        "wanting to hurt myself or commit suicide",
        "i hate myself",
        "i am unloved"
    ],
    "illegal": [
        "planning illegal activities or evading law enforcement",
        "i have illegal items in my possession",
        "how to commit a crime without getting caught"
    ],
    "concerning_context": [
        "interest in crowded public places",
        "questions about schools or locations with many people",
        "focus on children or vulnerable groups in real-world settings",
        "asking about large events or gathering locations"
    ]
}

def score_category(msg_emb, anchor_embs):
    return max(cosine_similarity(msg_emb, a) for a in anchor_embs)

anchors = {
    k: [get_embedding(t) for t in texts]
    for k, texts in anchor_texts.items()
}

danger_state = {k: 0.0 for k in anchors}
DECAY = 0.8

history = []
MAX_HISTORY = 3

print("Type messages (Ctrl+C to quit)\n")

while True:
    user_input = input("You: ")
    msg_emb = get_embedding(user_input)

    print("\n--- Risk Analysis ---")

    current_scores = {}

    for k, anchor_embs in anchors.items():
        sim = score_category(msg_emb, anchor_embs)
        sim = max(0, sim)

        danger_state[k] = DECAY * danger_state[k] + sim
        current_scores[k] = sim

        print(f"{k:20s} | similarity: {sim:.3f} | running: {danger_state[k]:.3f}")

    history.append(current_scores)
    if len(history) > MAX_HISTORY:
        history.pop(0)

    violence = danger_state["violence"]
    context = danger_state["concerning_context"]

    interaction_bonus = context * violence * 2

    escalation_bonus = 0

    if len(history) >= 2:
        prev_context = history[-2]["concerning_context"]
        curr_violence = history[-1]["violence"]

        if prev_context > 0.25 and curr_violence > 0.2:
            escalation_bonus += 1.5

    total_risk = sum(danger_state.values()) + interaction_bonus + escalation_bonus

    print(f"\nInteraction Bonus: {interaction_bonus:.3f}")
    print(f"Escalation Bonus: {escalation_bonus:.3f}")
    print(f"TOTAL DANGER SCORE: {total_risk:.3f}")

    if total_risk > 2.5:
        print("HIGH RISK\n")
    elif total_risk > 1.5:
        print("MEDIUM RISK\n")
    else:
        print("LOW RISK\n")