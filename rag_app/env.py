import os

from dotenv import load_dotenv


def load_environment() -> None:
    dotenv_path = os.getenv("DOTENV_PATH")
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)
    else:
        load_dotenv(override=False)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def get_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def get_missing_config() -> list[str]:
    missing = []
    if not get_env("DASHSCOPE_API_KEY", "dashscope_api_key"):
        missing.append("DASHSCOPE_API_KEY")
    if not get_env("DEEPSEEK_API_KEY"):
        missing.append("DEEPSEEK_API_KEY")
    return missing


def get_dashscope_api_key() -> str:
    dashscope_api_key = get_env("DASHSCOPE_API_KEY", "dashscope_api_key")
    if not dashscope_api_key:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY，无法创建向量索引。")
    return dashscope_api_key


def get_deepseek_api_key() -> str:
    deepseek_api_key = get_env("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY，无法生成回答。")
    return deepseek_api_key
