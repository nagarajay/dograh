"use client";

import "@xyflow/react/dist/style.css";

import {
    Background,
    BackgroundVariant,
    ReactFlow,
    type ReactFlowInstance,
} from "@xyflow/react";
import { useMemo, useRef } from "react";

import { WorkflowProvider } from "@/app/workflow/[workflowId]/contexts/WorkflowContext";
import { layoutNodes } from "@/app/workflow/[workflowId]/utils/layoutNodes";
import CustomEdge from "@/components/flow/edges/CustomEdge";
import { GenericNode } from "@/components/flow/nodes/GenericNode";
import { useNodeSpecs } from "@/components/flow/renderer";
import { type FlowEdge, type FlowNode, NodeType } from "@/components/flow/types";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

import {
    type InspectedDocument,
    inspectedDocumentsAsCatalog,
    type InspectedEdge,
    inspectedEdgesToFlowEdges,
    type InspectedNode,
    inspectedNodesToFlowNodes,
    type InspectedTool,
    inspectedToolsAsCatalog,
} from "./agentFlow";

const edgeTypes = { custom: CustomEdge };

interface Props {
    nodes: InspectedNode[];
    edges: InspectedEdge[];
    tools: InspectedTool[];
    documents: InspectedDocument[];
    /** "draft" / "published" / "legacy_workflow_definition". */
    inspectedSource: string;
}

/**
 * The customer's agent graph, rendered read-only for a super admin.
 *
 * Reuses the editor's own canvas — the same node renderer, edge renderer and
 * dagre layout — so what a super admin sees is what the customer built, not a
 * second drawing of it that can drift. It is strictly a viewer: the canvas is
 * not draggable or connectable, and the workflow context is marked read-only,
 * which suppresses every edit affordance and the stale-reference cleanup those
 * components would otherwise write back. There is no super-admin write path to
 * a customer's definition, so an edit here could only fail; not offering it is
 * the honest behaviour.
 */
export function AgentWorkflowGraph({
    nodes,
    edges,
    tools,
    documents,
    inspectedSource,
}: Props) {
    const { specs } = useNodeSpecs();
    const rfInstance = useRef<ReactFlowInstance<FlowNode, FlowEdge> | null>(null);

    const flowNodes = useMemo(() => {
        const built = inspectedNodesToFlowNodes(nodes);
        const builtEdges = inspectedEdgesToFlowEdges(edges);
        // The inspection payload carries no positions, so lay the graph out the
        // way the editor's "auto layout" would.
        return layoutNodes(built, builtEdges, "TB", rfInstance);
    }, [edges, nodes]);

    const flowEdges = useMemo(() => inspectedEdgesToFlowEdges(edges), [edges]);

    const nodeTypes = useMemo(() => {
        const typeNames = new Set<string>([
            ...Object.values(NodeType),
            ...specs.map((spec) => spec.name),
            ...flowNodes.map((node) => node.type),
        ]);
        return Object.fromEntries(
            Array.from(typeNames).map((typeName) => [typeName, GenericNode]),
        );
    }, [flowNodes, specs]);

    const workflowContext = useMemo(
        () => ({
            // No super-admin write path exists; read-only mode never calls this.
            saveWorkflow: async () => {},
            tools: inspectedToolsAsCatalog(tools),
            documents: inspectedDocumentsAsCatalog(documents),
            recordings: [],
            readOnly: true,
        }),
        [documents, tools],
    );

    return (
        <Card>
            <CardHeader>
                <CardTitle>Agent graph</CardTitle>
                <CardDescription>
                    Read-only view of the {inspectedSource} definition — the same canvas the
                    customer edits. Double-click a node to inspect its configuration.
                </CardDescription>
            </CardHeader>
            <CardContent>
                {flowNodes.length === 0 ? (
                    <p className="text-sm text-muted-foreground">
                        This agent has no nodes defined.
                    </p>
                ) : (
                    <div className="h-[600px] w-full rounded-md border">
                        <WorkflowProvider value={workflowContext}>
                            <ReactFlow
                                nodes={flowNodes}
                                edges={flowEdges}
                                nodeTypes={nodeTypes}
                                edgeTypes={edgeTypes}
                                minZoom={0.2}
                                nodesDraggable={false}
                                nodesConnectable={false}
                                edgesReconnectable={false}
                                zoomOnDoubleClick={false}
                                deleteKeyCode={null}
                                onInit={(instance) => {
                                    rfInstance.current = instance;
                                    setTimeout(() => {
                                        instance.fitView({
                                            padding: 0.2,
                                            duration: 200,
                                            maxZoom: 0.75,
                                        });
                                    }, 0);
                                }}
                            >
                                <Background
                                    variant={BackgroundVariant.Dots}
                                    gap={16}
                                    size={1}
                                    color="#94a3b8"
                                />
                            </ReactFlow>
                        </WorkflowProvider>
                    </div>
                )}
            </CardContent>
        </Card>
    );
}
