# CAP General Architecture Diagrams

This directory contains architecture diagrams generated from the unit-test examples and current core modules.

## Images

- [core_class_relationship.svg](core_class_relationship.svg): core class relationships.
- [local_execution_flow.svg](local_execution_flow.svg): local Franka example execution flow.
- [skills_mcp_call_flow.svg](skills_mcp_call_flow.svg): skills, MCP server, and CAP runtime call flow.
- [policy_graph_execution.svg](policy_graph_execution.svg): policy graph/operator execution flow.

## Sources

The Mermaid source files are kept next to the rendered images:

- [core_class_relationship.mmd](core_class_relationship.mmd)
- [local_execution_flow.mmd](local_execution_flow.mmd)
- [skills_mcp_call_flow.mmd](skills_mcp_call_flow.mmd)
- [policy_graph_execution.mmd](policy_graph_execution.mmd)

Regenerate the SVG files with:

```bash
mmdc -p doc/mermaid-puppeteer-config.json -i doc/core_class_relationship.mmd -o doc/core_class_relationship.svg
mmdc -p doc/mermaid-puppeteer-config.json -i doc/local_execution_flow.mmd -o doc/local_execution_flow.svg
mmdc -p doc/mermaid-puppeteer-config.json -i doc/skills_mcp_call_flow.mmd -o doc/skills_mcp_call_flow.svg
mmdc -p doc/mermaid-puppeteer-config.json -i doc/policy_graph_execution.mmd -o doc/policy_graph_execution.svg
```
