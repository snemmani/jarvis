_TG_MAX = 4096


async def send_long(reply_func, text: str, **kwargs) -> None:
    """Send text split across multiple messages if it exceeds Telegram's 4096-char limit."""
    while text:
        if len(text) <= _TG_MAX:
            await reply_func(text, **kwargs)
            return
        split_at = text.rfind("\n", 0, _TG_MAX)
        if split_at <= 0:
            split_at = _TG_MAX
        await reply_func(text[:split_at], **kwargs)
        text = text[split_at:].lstrip("\n")
