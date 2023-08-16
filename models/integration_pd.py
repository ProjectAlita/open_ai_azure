from pydantic import BaseModel

from tools import session_project
from pylon.core.tools import log
from ...integrations.models.pd.integration import SecretField
import openai


class IntegrationModel(BaseModel):
    api_token: SecretField | str
    model_name: str = 'gpt-35-turbo'
    temperature: float = 0
    api_version: str = '2023-03-15-preview'
    api_base: str = "https://ai-proxy.lab.epam.com"

    def check_connection(self):
        openai.api_key = self.api_token.unsecret(session_project.get())
        openai.api_type = "azure"
        openai.api_version = self.api_version
        openai.api_base = self.api_base
        try:
            openai.Model.list()
        except Exception as e:
            log.error(e)
            return str(e)
        return True
