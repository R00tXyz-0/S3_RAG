VISION_PROMPT = """Analyze this course slide for a RAG knowledge base.

Extract all useful information contained in the visual content:

- readable text
- definitions
- technical concepts
- Oracle / PL-SQL terminology
- SQL and PL-SQL code exactly as visible
- tables and their values
- diagram labels
- relationships represented by diagrams
- examples and commands

Preserve technical terminology exactly.
Preserve SQL/PL-SQL syntax as accurately as possible.
Do not invent information.
Do not describe decorative elements, logos, colors, or page design.
If some text is unreadable, explicitly indicate that it is unreadable.

Return only useful textual knowledge that can be merged into the native PDF text."""


def build_vision_prompt(source, chapter, slide_title):
    ctx_lines = ["Context:"]
    if source:
        ctx_lines.append(f"Document: {source}")
    if chapter:
        ctx_lines.append(f"Chapter: {chapter}")
    if slide_title:
        ctx_lines.append(f"Slide title: {slide_title}")
    ctx_lines.append("")
    return "\n".join(ctx_lines) + VISION_PROMPT
