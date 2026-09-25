from eipg.simulators.huggingface_local import HuggingFaceLocalChatClient, HuggingFaceLocalConfig


def test_hf_local_config_from_config():
    cfg = HuggingFaceLocalConfig.from_config(
        {
            "model": "google/gemma-3-12b-it",
            "dtype": "bfloat16",
            "device_map": "auto",
            "max_new_tokens": 64,
            "do_sample": False,
            "temperature": None,
            "cache_dir": None,
        }
    )
    assert cfg.model == "google/gemma-3-12b-it"
    assert cfg.max_new_tokens == 64
    assert cfg.do_sample is False
    assert cfg.cache_dir is None


def test_structured_messages_are_gemma_compatible():
    messages = [
        {"role": "system", "content": "Return JSON only."},
        {"role": "user", "content": "Choose one."},
    ]
    structured = HuggingFaceLocalChatClient._structured_messages(messages)
    assert structured == [
        {
            "role": "system",
            "content": [{"type": "text", "text": "Return JSON only."}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Choose one."}],
        },
    ]
