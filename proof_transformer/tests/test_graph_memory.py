from __future__ import annotations

from common import BrainTestCase
from proof_transformer.memory.graph import GraphMemory


class GraphMemoryTests(BrainTestCase):
    def test_create_and_expand_graph(self) -> None:
        graph = GraphMemory(self.conn)
        node_a = graph.create_node("project_fact", "Brain root", "The root is E:/AUSTIN_AI_BRAIN.")
        node_b = graph.upsert_node_by_name_kind("project_fact", "Ledger", "The ledger is hash-linked.")
        edge = graph.create_edge(node_a["id"], node_b["id"], "supports", weight=0.8, confidence=0.9)

        fetched = graph.get_node(node_a["id"])
        self.assertEqual(fetched["name"], "Brain root")
        self.assertEqual(len(graph.get_edges_for_node(node_a["id"])), 1)

        subgraph = graph.expand_subgraph([node_a["id"]], max_hops=2, minimum_weight=0.5, confidence_threshold=0.5)
        self.assertEqual({node["id"] for node in subgraph.nodes}, {node_a["id"], node_b["id"]})
        self.assertEqual(len(subgraph.edges), 1)

        strengthened = graph.strengthen_edge(edge["id"], amount=0.1)
        weakened = graph.weaken_edge(edge["id"], amount=0.2)
        self.assertGreater(strengthened["weight"], edge["weight"])
        self.assertLess(weakened["weight"], strengthened["weight"])
        self.assertTrue(graph.search_nodes_keyword("ledger"))
