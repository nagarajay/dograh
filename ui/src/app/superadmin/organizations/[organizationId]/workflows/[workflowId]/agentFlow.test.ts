import { describe, expect, it } from "vitest";

import {
    inspectedDocumentsAsCatalog,
    inspectedEdgesToFlowEdges,
    inspectedNodesToFlowNodes,
    inspectedToolsAsCatalog,
} from "./agentFlow";

describe("inspectedNodesToFlowNodes", () => {
    it("carries narrative fields, name and references onto the canvas node", () => {
        const [node] = inspectedNodesToFlowNodes([
            {
                id: "n1",
                type: "agentNode",
                name: "Qualify",
                narrative: { prompt: "Ask for the account number", allow_interrupt: true },
                tool_uuids: ["tool-1"],
                document_uuids: ["doc-1"],
                mcp_tool_filters: { "tool-1": ["search"] },
            },
        ]);

        expect(node.type).toBe("agentNode");
        expect(node.data.name).toBe("Qualify");
        expect(node.data.prompt).toBe("Ask for the account number");
        expect(node.data.allow_interrupt).toBe(true);
        expect(node.data.tool_uuids).toEqual(["tool-1"]);
        expect(node.data.document_uuids).toEqual(["doc-1"]);
        expect(node.data.mcp_tool_filters).toEqual({ "tool-1": ["search"] });
    });

    it("falls back to the node id when the definition has no name", () => {
        const [node] = inspectedNodesToFlowNodes([
            {
                id: "n2",
                type: null,
                name: null,
                narrative: {},
                tool_uuids: [],
                document_uuids: [],
            },
        ]);

        expect(node.data.name).toBe("n2");
        expect(node.type).toBe("agentNode");
    });
});

describe("inspectedEdgesToFlowEdges", () => {
    it("gives parallel transitions distinct ids and drops dangling ones", () => {
        const edges = inspectedEdgesToFlowEdges([
            { source: "a", target: "b", label: "yes", condition: "answered" },
            { source: "a", target: "b", label: "no", condition: "declined" },
            { source: "a", target: null, label: "nowhere", condition: null },
        ]);

        expect(edges).toHaveLength(2);
        expect(new Set(edges.map((edge) => edge.id)).size).toBe(2);
        expect(edges[0].data).toEqual({ label: "yes", condition: "answered" });
        expect(edges[0].type).toBe("custom");
    });
});

describe("catalogs", () => {
    it("keeps resolved references only, so unresolved ones are not invented", () => {
        const tools = inspectedToolsAsCatalog([
            {
                tool_uuid: "t1",
                name: "Lookup",
                description: null,
                category: "http",
                status: "active",
                resolved: true,
            },
            {
                tool_uuid: "t2",
                name: null,
                description: null,
                category: null,
                status: null,
                resolved: false,
            },
        ]);
        const documents = inspectedDocumentsAsCatalog([
            {
                document_uuid: "d1",
                filename: "policy.pdf",
                retrieval_mode: "full",
                processing_status: "completed",
                total_chunks: 3,
                resolved: true,
            },
            {
                document_uuid: "d2",
                filename: null,
                retrieval_mode: null,
                processing_status: null,
                total_chunks: null,
                resolved: false,
            },
        ]);

        expect(tools.map((tool) => tool.tool_uuid)).toEqual(["t1"]);
        expect(tools[0].name).toBe("Lookup");
        expect(documents.map((document) => document.document_uuid)).toEqual(["d1"]);
        expect(documents[0].filename).toBe("policy.pdf");
    });
});
