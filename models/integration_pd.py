from typing import List
from pydantic import BaseModel, root_validator, validator

from tools import session_project, rpc_tools
from pylon.core.tools import log
from ...integrations.models.pd.integration import SecretField


class CapabilitiesModel(BaseModel):
    completion: bool = False
    chat_completion: bool = True
    embeddings: bool = False


class AIModel(BaseModel):
    id: str
    name: str
    capabilities: CapabilitiesModel = CapabilitiesModel()

    @validator('name', always=True, check_fields=False)
    def name_validator(cls, value, values):
        return values.get('model', value)

class IntegrationModel(BaseModel):
    api_token: SecretField | str
    model_name: str = 'gpt-35-turbo'
    models: List[AIModel] = []
    api_version: str = '2023-03-15-preview'
    api_base: str = "https://ai-proxy.lab.epam.com"
    api_type: str = "azure"
    temperature: float = 0
    max_tokens: int = 7
    top_p: float = 0.8
    stream: bool = False

    @root_validator(pre=True)
    def prepare_model_list(cls, values):
        models = values.get('models')
        if models and isinstance(models[0], str):
            values['models'] = [AIModel(id=model, name=model).dict(by_alias=True) for model in models]
        return values

    def check_connection(self):
        import openai
        openai.api_key = self.api_token.unsecret(session_project.get())
        openai.api_type = self.api_type
        openai.api_version = self.api_version
        openai.api_base = self.api_base
        try:
            openai.Model.list()
        except Exception as e:
            log.error(e)
            return str(e)
        return True

    def refresh_models(self, project_id):
        integration_name = 'open_ai_azure'
        payload = {
            'name': integration_name,
            'settings': self.dict(),
            'project_id': project_id
        }
        return getattr(rpc_tools.RpcMixin().rpc.call, f'{integration_name}_set_models')(payload)


class AzureOpenAISettings(BaseModel):
    model_name: str = 'gpt-35-turbo'
    api_version: str = '2023-03-15-preview'
    api_base: str = "https://ai-proxy.lab.epam.com"
    temperature: float = 0
    max_tokens: int = 7
    top_p: float = 0.8
