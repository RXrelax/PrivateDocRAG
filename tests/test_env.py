import rag_app.env as app_env


def test_load_environment_preserves_injected_values(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(app_env, "load_dotenv", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setenv("DOTENV_PATH", "/tmp/example.env")
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "FALSE")

    app_env.load_environment()

    assert calls == [(('/tmp/example.env',), {"override": False})]
    assert app_env.os.environ["KMP_DUPLICATE_LIB_OK"] == "FALSE"
