import type { DocumentResponseSchema, ToolResponse } from "@/client/types.gen";
import type { FlowEdge, FlowNode } from "@/components/flow/types";

/**
 * Translation from the super-admin agent inspection payload into the shapes
 * the existing workflow canvas renders.
 *
 * The inspection endpoint reports behaviour, not layout: it has no node
 * positions, so positions are computed on the client with the editor's own
 * dagre layout. Nothing here reads or invents fields the endpoint did not
 * return — masked values stay masked, and a node type the canvas does not know
 * still renders through the generic node.
 */

export interface InspectedNode {
    id: string;
    type?: string | null;
    name?: string | null;
    narrative: Record<string, unknown>;
    tool_uuids: string[];
    document_uuids: string[];
    mcp_tool_filters?: Record<string, unknown> | null;
}

export interface InspectedEdge {
    source?: string | null;
    target?: string | null;
    label?: string | null;
    condition?: string | null;
}

export interface InspectedTool {
    tool_uuid: string;
    name?: string | null;
    description?: string | null;
    category?: string | null;
    status?: string | null;
    resolved: boolean;
}

export interface InspectedDocument {
    document_uuid: string;
    filename?: string | null;
    retrieval_mode?: string | null;
    processing_status?: string | null;
    total_chunks?: number | null;
    resolved: boolean;
}

export function inspectedNodesToFlowNodes(nodes: InspectedNode[]): FlowNode[] {
    return nodes.map((node) => ({
        id: node.id,
        type: node.type || "agentNode",
        position: { x: 0, y: 0 },
        data: {
            ...node.narrative,
            name: node.name || node.id,
            tool_uuids: node.tool_uuids,
            document_uuids: node.document_uuids,
            mcp_tool_filters:
                (node.mcp_tool_filters as Record<string, string[]> | null) ?? undefined,
        },
    }));
}

/**
 * Edges arrive without ids (the inspection payload describes transitions, not
 * canvas elements), so ids are synthesised. The index keeps parallel edges
 * between the same two nodes distinct, which the canvas needs to offset them.
 */
export function inspectedEdgesToFlowEdges(edges: InspectedEdge[]): FlowEdge[] {
    return edges
        .filter((edge) => edge.source && edge.target)
        .map((edge, index) => ({
            id: `${edge.source}-${edge.target}-${index}`,
            source: edge.source as string,
            target: edge.target as string,
            type: "custom",
            data: {
                label: edge.label ?? "",
                condition: edge.condition ?? "",
            },
        }));
}

/**
 * Tool and document badges on the canvas resolve names from the workflow
 * context's catalogs. The super admin has no access to the customer's tool or
 * knowledge-base endpoints, so the catalogs are built from what the inspection
 * already reported — the same masked, catalog-only records shown in the tables
 * below the graph. Unresolved references are left out, so they render as
 * missing rather than as a fabricated entry.
 */
export function inspectedToolsAsCatalog(tools: InspectedTool[]): ToolResponse[] {
    return tools
        .filter((tool) => tool.resolved)
        .map((tool) => ({
            id: 0,
            tool_uuid: tool.tool_uuid,
            name: tool.name ?? tool.tool_uuid,
            description: tool.description ?? null,
            category: tool.category ?? "",
            icon: null,
            icon_color: null,
            status: tool.status ?? "",
            definition: {},
            created_at: "",
            updated_at: null,
        }));
}

export function inspectedDocumentsAsCatalog(
    documents: InspectedDocument[],
): DocumentResponseSchema[] {
    return documents
        .filter((document) => document.resolved)
        .map((document) => ({
            id: 0,
            document_uuid: document.document_uuid,
            filename: document.filename ?? document.document_uuid,
            file_size_bytes: 0,
            file_hash: "",
            mime_type: "",
            processing_status: document.processing_status ?? "",
            total_chunks: document.total_chunks ?? 0,
            retrieval_mode: document.retrieval_mode ?? undefined,
            custom_metadata: {},
            docling_metadata: {},
            created_at: "",
            updated_at: "",
            organization_id: 0,
            created_by: 0,
            is_active: true,
        }));
}
