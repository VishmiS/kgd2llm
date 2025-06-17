# query_graph.py

import pickle
import networkx as nx
import re


class KnowledgeGraphQuery:
    def __init__(self, graph: nx.DiGraph):
        self.G = graph

    def get_capital_of_country(self, country_name: str):
        if country_name in self.G:
            for neighbor in self.G.successors(country_name):
                if self.G[country_name][neighbor].get('relation') == 'has_capital':
                    return neighbor
        return None

    def get_country_of_capital(self, capital_name: str):
        if capital_name in self.G:
            for neighbor in self.G.successors(capital_name):
                if self.G[capital_name][neighbor].get('relation') == 'is_capital_of':
                    return neighbor
        return None

    def get_population(self, name: str):
        if name in self.G:
            return self.G.nodes[name].get('population')
        return None

    def list_all_countries_and_capitals(self):
        result = []
        for node, data in self.G.nodes(data=True):
            if data.get('type') == 'Country':
                capital = self.get_capital_of_country(node)
                result.append((node, capital))
        return result


def answer_query(question: str, graph: nx.DiGraph) -> str:
    question_lower = question.lower()
    if "capital of" in question_lower:
        match = re.search(r"capital of ([a-zA-Z\s]+)", question_lower)
        if match:
            country = match.group(1).strip().title()
            if country in graph:
                for neighbor in graph.successors(country):
                    if graph.nodes[neighbor].get('is_capital'):
                        return f"The capital of {country} is {neighbor}."
                return f"Sorry, capital of {country} not found."
            else:
                return f"{country} is not in the graph."
    return "Sorry, I couldn't understand the question."


def load_graph(path: str) -> nx.DiGraph:
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    graph_path = "/root/pycharm_semanticsearch/knowledge_graph/graph.pkl"
    G = load_graph(graph_path)
    query = KnowledgeGraphQuery(G)

    print("Example Queries:")
    print("Capital of Germany:", query.get_capital_of_country("Germany"))
    print("Country of Berlin:", query.get_country_of_capital("Berlin"))
    print("Population of United States:", query.get_population("United States"))


    print("\nNatural Language Queries:")
    questions = [
        "What is the capital of Germany?",
        "Tell me the capital of France.",
        "What is the capital of Canada?"
    ]
    for q in questions:
        print(f"Q: {q}")
        print("A:", answer_query(q, G))
        print()


if __name__ == "__main__":
    main()
