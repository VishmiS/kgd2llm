# build_graph.py

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from typing import Tuple


def load_country_city_data(csv_path: str) -> pd.DataFrame:
    """Load country-city data from a CSV file."""
    df = pd.read_csv(csv_path, sep=',')
    print(f"Loaded CSV with columns: {df.columns.tolist()}")
    return df


def create_country_capital_graph(df: pd.DataFrame) -> nx.DiGraph:
    """Create a directed graph of countries and their capitals."""
    G = nx.DiGraph()

    for _, row in df.iterrows():
        country = row['countryLabel']
        country_id = row['country']
        country_pop = int(row['countryPopulation'])

        capital = row['capitalLabel']
        capital_id = row['capital']
        capital_pop = int(row['capitalPopulation'])

        G.add_node(country, type='Country', wikidata_id=country_id, population=country_pop)
        G.add_node(capital, type='City', wikidata_id=capital_id, population=capital_pop, is_capital=True)

        G.add_edge(country, capital, relation='has_capital')
        G.add_edge(capital, country, relation='is_capital_of')

    return G


def visualize_graph(
    G: nx.DiGraph,
    save_path: str = None,
    figsize: Tuple[int, int] = (16, 12),
    node_size: int = 1200,
    font_size: int = 10,
) -> None:
    """Visualize the graph."""
    pos = nx.spring_layout(G, k=0.4, iterations=50)
    plt.figure(figsize=figsize)

    node_colors = ['skyblue' if data['type'] == 'Country' else 'lightgreen' for _, data in G.nodes(data=True)]

    nx.draw(G, pos, with_labels=True, node_size=node_size, font_size=font_size,
            node_color=node_colors, edge_color='gray')

    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)

    plt.title("Country-Capital Knowledge Graph")
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Graph visualization saved to {save_path}")

    plt.show()


def save_graph(G: nx.DiGraph, path: str) -> None:
    """Save the graph to a file using pickle."""
    with open(path, 'wb') as f:
        pickle.dump(G, f)
    print(f"Graph saved to {path}")


def main():
    csv_path = "/root/pycharm_semanticsearch/dataset/wikidata_country_city.csv"
    image_path = "/root/pycharm_semanticsearch/knowledge_graph/output_graph.png"
    graph_pickle_path = "/root/pycharm_semanticsearch/knowledge_graph/graph.pkl"

    df = load_country_city_data(csv_path)
    G = create_country_capital_graph(df)

    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    visualize_graph(G, save_path=image_path)
    save_graph(G, graph_pickle_path)


if __name__ == "__main__":
    main()
