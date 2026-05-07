import random

from lcnc import *

def feature_drop_rate(prompt, fragments, generate_arch_fn, attributions):
    """
    Porównanie: top fragment vs random fragment
    """

    full_arch = generate_arch_fn(prompt)
    full_features = extract_feature_vector(full_arch)

    # ranking fragmentów (sumaryczny wpływ)
    def score(frag):
        return sum(abs(v) for v in attributions[frag].values())

    top_frag = max(fragments, key=score)
    random_frag = random.choice(fragments)
    bottom_frag = min(fragments, key=score)

    def drop(fragment):
        reduced = " ".join([f for f in fragments if f != fragment])
        arch = generate_arch_fn(reduced)
        features = extract_feature_vector(arch)

        diff = feature_diff(full_features, features)
        drop_count = sum(1 for v in diff.values() if v > 0)
        return drop_count

    return {
        "top_fragment": drop(top_frag),
        "random_fragment": drop(random_frag),
        "bottom_fragment": drop(bottom_frag),
    }

if __name__ == "__main__":
    user_prompt = """
    The system should allow users to register and log in.
    It must also provide a dashboard for managing their profiles and settings.
    Additionally, the application should support offline access and send notifications for important updates.
    """

    fragments = extract_prompt_fragments(user_prompt)
    attributions = shap_sampling_attribution(fragments, generate_architecture, samples=5)
    results = feature_drop_rate(user_prompt, fragments, generate_architecture, attributions)
    print(results)

    # save results to a file
    with open("feature_drop_results.txt", "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")