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
    while random_frag == top_frag:
        random_frag = random.choice(fragments)
    # bottom_frag = min(fragments, key=score)

    def drop(fragment):
        reduced = " ".join([f for f in fragments if f != fragment])
        arch = generate_arch_fn(reduced)
        features = extract_feature_vector(arch)

        diff = feature_diff(full_features, features)
        drop_count = sum(1 for v in diff.values() if v > 0)
        return (fragment, drop_count)

    return {
        "top_fragment": drop(top_frag),
        "random_fragment": drop(random_frag),
        # "bottom_fragment": drop(bottom_frag),
    }

if __name__ == "__main__":
    user_prompt = """
    The system should allow users to register and log in with secure authentication.
    Each user will be able to create and manage their own tasks with priorities and deadlines.
    It must also provide a comprehensive dashboard for managing their profiles, settings, and preferences.
    Additionally, the application should support offline access and send notifications for important updates.
    The system should include role-based access control for different user types.
    Data synchronization across multiple devices should be seamless and automatic.
    The application must provide a clean and intuitive user interface with dark mode support.
    All user data should be encrypted and securely stored in the cloud database.
    The system should allow users to collaborate on shared projects and tasks.
    Analytics and reporting features should help users track their productivity and progress.
    """

    user_prompt2 = """
    The application should provide user authentication and task management capabilities.
    A dashboard for profile and settings management is essential.
    The system must support offline access and real-time notifications.
    Role-based access control and cloud data synchronization are required features.
    """

    fragments = extract_prompt_fragments(user_prompt2)
    attributions = shap_sampling_attribution(fragments, generate_architecture, samples=5)
    with open("ex1_fragments2.txt", "w") as f:
        print("Fragmenty i ich wpływ:")
        for i, frag in enumerate(fragments, 1):
            score = sum(abs(v) for v in attributions[frag].values())
            print(f"{i}. {frag} (Score: {score})")
            f.write(f"{i}. {frag} (Score: {score})\n")

    results = feature_drop_rate(user_prompt2, fragments, generate_architecture, attributions)
    print(results)

    # save results to a file
    with open("ex1_feature_drop_results2.txt", "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")