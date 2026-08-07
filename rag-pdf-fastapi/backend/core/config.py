from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    API_KEY: str

    MODEL_NAME: str = "all-MiniLM-L6-v2"

    URL: str = "https://openrouter.ai/api/v1"
    PDF_PATH: str = "data/sample.pdf"
    INDEX_PATH: str = "data/index.faiss"
    CHUNKS_PATH: str = "data/chunks.pkl"

    CHUNK_SIZE: int = 500
    OVERLAP: int = 100
    TOP_K: int = 3

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

    CHAT_MODEL: str = "gpt-3.5-turbo"


settings = Settings()