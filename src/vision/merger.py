def merge_texts(native_text, visual_text):
    native = (native_text or "").strip()
    visual = (visual_text or "").strip()
    if native and visual:
        return native + "\n\n" + visual
    return native or visual
