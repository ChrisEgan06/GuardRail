import numpy as np
from openai import OpenAI
import matplotlib.pyplot as plt

# Any similarity bellow this is regarded as random and disgarded
NOISE_FLOOR = 0.18

# Base thresholds for categories (not heavily used in this version)
V_THRESHOLD = 0.2
W_THRESHOLD = 0.2
C_THRESHOLD = 0.2

#Thresholds for if a signal is strong enough to trigger escelation across messages
CROSS_V_THRESHOLD = 0.2
CROSS_W_THRESHOLD = 0.2
CROSS_C_THRESHOLD = 0.25
CROSS_S_THRESHOLD = 0.2

#control memory over time
DECAY_DEFAULT = 0.7
DECAY_SLOW = 0.9

#only keep past 3 messsages for cross analysis
MAX_HISTORY = 3

# scales interaction bonus
INTERACTION_SCALE = 1.0

#giant spike added for if escelation is spotted
CROSS_BONUS = 2.0

#print internalo values
DEBUG = True


parser = argparse.ArgumentParser()
parser.add_argument("--no_plot", action="store_true")
args = parser.parse_args()

use_plot = not args.no_plot

client = OpenAI()

#semantic meaning embedding
def get_embedding(text):
    """
    Converts text into a vector (high-dimensional numeric representation).
    
    Each sentence becomes a vector like:
    [0.12, -0.93, 0.44, ..., 0.01]
    
    Similar meaning → similar vectors in space
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)

#vector similarity
def cosine_similarity(a, b):
    """
    Computes similarity between two vectors using cosine similarity.
    
    Formula:
        cos(theta) = (a · b) / (||a|| * ||b||)

    Intuition:
        1.0 = identical direction (very similar meaning)
        0.0 = unrelated
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

#Anchors defining semantic regions in vector space
negative_anchor_texts = {
    "violence": [
        "planning a violent attack",
        "hurting people intentionally",
        "attacking someone",
        "is doing this lethal"
    ],
    "weaponry": [
        "guns and firearms",
        "weapon safety and handling",
        "types of guns like glock or ar-15",
        "how weapons function",
        "using firearms"
    ],
    "self_harm": [
        "wanting to hurt myself",
        "i hate my life",
        "i want my life to end",
        "i hate myself"
    ],
    "illegal": [
        "planning illegal activities"
    ],
    "concerning_context": [
        "schools or public places",
        "crowded areas",
        "busy public locations",
        "when is blank busiest"
    ]
}

positive_anchor_texts = {
    "positive": [
        "i love myself",
        "my life is fantastic",
        "i want to spread positivity",
        "i love everybody",
        "everybody loves me"
    ]
}

def score_category(msg_emb, anchor_embs):
    """
    Compare message embedding to ALL anchor embeddings in a category.
    
    We take the MAX similarity    
    
    score = max(cosine_similarity(msg, anchor_i)) over all anchors
    """
    return max(cosine_similarity(msg_emb, a) for a in anchor_embs)

#precompute anchor embeddings
negative_anchor_embs = {
    k: [get_embedding(t) for t in texts]
    for k, texts in negative_anchor_texts.items()
}

positive_anchor_embs = {
    l: [get_embedding(t2) for t2 in texts]
    for l, texts in positive_anchor_texts.items()
}

#STATE

#store running risk
danger_state = {k: 0.0 for k in negative_anchor_embs}  # FIX: correct variable name
# Stores running (decayed) risk per category

#store messages for temporal reasoning
history = []

#plotting utils
risk_over_time = []
turns = []

print("Type messages (Ctrl+C to quit)\n")

step = 0

try:
    while True:
        user_input = input("You: ")
        msg_emb = get_embedding(user_input)

        print("\n--- Risk Analysis ---")

        current_scores = {}

        #Base schoring
        for k, neg_emb_list in negative_anchor_embs.items():  # FIX: correct loop variable
            raw_sim = score_category(msg_emb, neg_emb_list)

            pos_score = 0 

            for j, pos_emb_list in positive_anchor_embs.items():
                pos_score = max(pos_score, score_category(msg_emb, pos_emb_list))


            if pos_score > 0.75:
                raw_sim = max(0, raw_sim - pos_score * 0.8) -

            # Apply noise filter
            sim = raw_sim if raw_sim > NOISE_FLOOR else 0

            #Select decary rate for catagory
            if k in ["weaponry", "concerning_context", "self_harm"]:
                decay = DECAY_SLOW
            else:
                decay = DECAY_DEFAULT

            # Temporal update:
            # new_value = decay * old_value + current_signal
            danger_state[k] = decay * danger_state[k] + sim

            current_scores[k] = sim

            print(f"{k:20s} | raw: {raw_sim:.3f} | sim: {sim:.3f} | running: {danger_state[k]:.3f}")

        # Store current step
        history.append(current_scores)
        if len(history) > MAX_HISTORY:
            history.pop(0)

        # Extract running signals
        v = danger_state["violence"]
        w = danger_state["weaponry"]
        c = danger_state["concerning_context"]
        s = danger_state["self_harm"]

        # Extract current message signals
        curr_v = current_scores["violence"]
        curr_w = current_scores["weaponry"]
        curr_c = current_scores["concerning_context"]
        curr_s = current_scores["self_harm"]


        #interaction bonus
        interaction_bonus = 0

        # Only trigger if current message has weapon/violence
        if curr_v > 0 or curr_w > 0:
            # (v + w) * c → combines capability + environment
            interaction_bonus = (v + w) * c * INTERACTION_SCALE

        if curr_s > 0:
            interaction_bonus += s * c * INTERACTION_SCALE

        #Cross turn logic
        cross_bonus = 0

        if len(history) >= 2:
            prev = history[-2]

            prev_v = prev["violence"]
            prev_w = prev["weaponry"]
            prev_c = prev["concerning_context"]
            prev_s = prev["self_harm"]

            if DEBUG:
                print("\n--- DEBUG CROSS ---")
                print(f"prev_v: {prev_v:.3f}, prev_w: {prev_w:.3f}, prev_c: {prev_c:.3f}, prev_s: {prev_s:.3f}")
                print(f"curr_v: {curr_v:.3f}, curr_w: {curr_w:.3f}, curr_c: {curr_c:.3f}, curr_s: {curr_s:.3f}")

            # CONTEXT → WEAPONRY
            if prev_c > CROSS_C_THRESHOLD and curr_w > CROSS_W_THRESHOLD:
                cross_bonus += CROSS_BONUS
                print("Triggered: CONTEXT → WEAPONRY")

            # WEAPONRY → CONTEXT
            if prev_w > CROSS_W_THRESHOLD and curr_c > CROSS_C_THRESHOLD:
                cross_bonus += CROSS_BONUS
                print("Triggered: WEAPONRY → CONTEXT")

            # SELF-HARM → CONTEXT
            if prev_s > CROSS_S_THRESHOLD and curr_c > CROSS_C_THRESHOLD:
                cross_bonus += CROSS_BONUS
                print("Triggered: SELF_HARM → CONTEXT")

            # CONTEXT → SELF-HARM
            if prev_c > CROSS_C_THRESHOLD and curr_s > CROSS_S_THRESHOLD:
                cross_bonus += CROSS_BONUS
                print("Triggered: CONTEXT → SELF_HARM")

        #FINAL SCORE
        total_risk = (
            sum(danger_state.values())  # accumulated category risk
            + interaction_bonus         # same-turn interaction
            + cross_bonus               # temporal escalation
        )

        risk_over_time.append(total_risk)
        turns.append(step)
        step += 1

        print(f"\nInteraction Bonus: {interaction_bonus:.3f}")
        print(f"Cross Bonus: {cross_bonus:.3f}")
        print(f"\nTOTAL DANGER SCORE: {total_risk:.3f}")

        if total_risk > 2.1:
            print("HIGH RISK\n")
        elif total_risk > 1.5:
            print("MEDIUM RISK\n")
        else:
            print("LOW RISK\n")

except KeyboardInterrupt:
    print("\nExiting...")


if use_plot and len(risk_over_time) > 0:
    plt.figure()

    plt.plot(turns, risk_over_time,marker='o', label="Risk Score")

    MEDIUM_THRESHOLD = 1.5
    HIGH_THRESHOLD = 2.5

    plt.axhline(y=MEDIUM_THRESHOLD,color='yellow', linestyle='--', label="Medium Risk Threshold")
    plt.axhline(y=HIGH_THRESHOLD, color='red', linestyle='--', label="High Risk Threshold")

    plt.xlabel("Conversation Turn")
    plt.ylabel("Risk Score")
    plt.title("Risk Over Time")

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.show()