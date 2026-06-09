def check_rules(user_text: str, assistant_text: str):
    user_l = (user_text or "").lower()
    assistant_l = (assistant_text or "").lower()

    if "sếp" in user_l:
        return False, "user_contains_assistant_call"

    if "tôi" not in assistant_l:
        return False, "assistant_missing_self"

    if "sếp" not in assistant_l:
        return False, "assistant_missing_call"

    return True, None
