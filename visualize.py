import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def plot_attribution_heatmap(attributions):
    """
    attributions:
    {
        fragment: {
            feature: score
        }
    }
    """

    # dataframe
    df = pd.DataFrame(attributions).T.fillna(0)

    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(df.values, aspect='auto')

    # labels
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))

    ax.set_xticklabels(df.columns, rotation=45, ha='right')
    ax.set_yticklabels(df.index)

    # wartości w komórkach
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            ax.text(
                j,
                i,
                f"{df.iloc[i, j]:.1f}",
                ha="center",
                va="center"
            )

    plt.title("Prompt-to-Feature Attribution Heatmap")
    plt.colorbar(im)
    plt.tight_layout()

    return fig

def plot_attribution_graph(attributions, threshold=0.1):

    G = nx.DiGraph()

    # dodaj node'y
    for frag in attributions:
        G.add_node(frag, bipartite=0)

        for feature, score in attributions[frag].items():

            if abs(score) >= threshold:
                G.add_node(feature, bipartite=1)
                G.add_edge(frag, feature, weight=score)

    # pos = nx.spring_layout(G, seed=42)

    edge_weights = [
        G[u][v]['weight']
        for u, v in G.edges()
    ]


    fragment_nodes = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 0]
    feature_nodes = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 1]

    pos = nx.bipartite_layout(G.to_undirected(), fragment_nodes)
    fig, ax = plt.subplots(figsize=(10, max(len(fragment_nodes)*2, len(feature_nodes) * 0.75)))

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=fragment_nodes,
        node_color="tab:blue",
        node_shape="o",
        node_size=3000,
        ax=ax,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=feature_nodes,
        node_color="tab:orange",
        node_shape="s",
        node_size=1000,
        ax=ax,
    )

    nx.draw_networkx_edges(
        G,
        pos,
        width=[abs(w) * 4 for w in edge_weights],
        edge_color=["tab:green" if w > 0 else "tab:red" for w in edge_weights],
        ax=ax,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=9,
        ax=ax,
    )

    ax.set_title("Prompt-to-Feature Attribution Graph")

    return fig

def compute_fragment_importance(attributions):

    importance = {}

    for frag, feats in attributions.items():
        importance[frag] = sum(abs(v) for v in feats.values())

    return importance

def plot_fragment_importance(attributions):

    importance = compute_fragment_importance(attributions)

    labels = list(importance.keys())
    values = list(importance.values())

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)

    plt.xticks(rotation=30, ha='right')
    plt.ylabel("Total Attribution")
    plt.title("Fragment Importance Ranking")

    plt.tight_layout()

    return plt.gcf()

def plot_stability(stability_scores):

    """
    stability_scores:
    {
        fragment: {
            feature: [scores]
        }
    }
    """

    fragments = []
    means = []
    stds = []

    for frag, values in stability_scores.items():

        flat = []

        for feat_scores in values.values():
            flat.extend(feat_scores)

        fragments.append(frag)
        means.append(np.mean(flat))
        stds.append(np.std(flat))

    plt.figure(figsize=(10, 5))

    plt.errorbar(
        fragments,
        means,
        yerr=stds,
        fmt='o'
    )

    plt.xticks(rotation=30, ha='right')
    plt.ylabel("Attribution")
    plt.title("Attribution Stability")

    plt.tight_layout()

    return plt.gcf()

if __name__ == "__main__":
    # Load attributions from a file
    import json

    with open("attributions.json", "r") as f:
        attributions = json.load(f)
    
    fig1 = plot_attribution_heatmap(attributions)
    fig1.savefig("attribution_heatmap.png")
    fig2 = plot_attribution_graph(attributions)
    fig2.savefig("attribution_graph.png")
    fig3 = plot_fragment_importance(attributions)
    fig3.savefig("fragment_importance.png")

    # Added random stability scores for demonstration
    stability_scores = {
        frag: {
            feat: [score + np.random.normal(0, 0.1) for _ in range(5)]
            for feat, score in feats.items()
        }
        for frag, feats in attributions.items()
    }
    fig4 = plot_stability(stability_scores)
    fig4.savefig("stability.png")