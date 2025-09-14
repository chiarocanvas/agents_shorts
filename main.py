import os
import yaml
from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIServerModel, ToolCollection
from mcp import StdioServerParameters
import fire  # type: ignore
load_dotenv()

def run():
    model_id = os.getenv("MODEL_ID", "ruadaptqwen3-8b-hybrid")

    os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:1234/v1")
    api_key = os.getenv("OPENAI_API_KEY", "lm-studio")

    client = OpenAIServerModel(
        model_id=model_id,
        api_key=api_key,
    )

    python_exe = os.environ.get("PYTHON_EXE", r"D:\proga\envs\aligment_norm\python.exe")
    mcp_script = os.path.abspath("mcp_tools.py")

    server_params = StdioServerParameters(
        command=python_exe,
        args=[mcp_script],
        env={**os.environ},
    )

    # Минимальный контракт инструментов (без навязывания плана)
    with  open("promts/promts.yaml", 'r', encoding="utf-8") as f:
        tools_guide =  yaml.safe_load(f)
        

    video_path = "https://www.youtube.com/watch?v=dfmTLtRhg0Q"
    task = f"""
    {tools_guide['tools']['system']}

    Цель: скачать видео по ссылке, транскрибировать, на основе транскрипта выбрать моменты и нарезать клипы. Источник: {video_path}
    """

    with ToolCollection.from_mcp(server_params, trust_remote_code=True) as tool_collection:
        agent = CodeAgent(
            tools=[*tool_collection.tools],
            model=client,
        )
        result = agent.run(task)
        print("✅ Результат:", result)


if __name__ == "__main__":
    fire.Fire(run)