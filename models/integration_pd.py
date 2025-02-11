import json
from typing import List, Optional

try:
    from pydantic.v1 import BaseModel, root_validator, validator
except:  # pylint: disable=W0702
    from pydantic import BaseModel, root_validator, validator

from tools import session_project, rpc_tools, VaultClient, worker_client, this, context
from pylon.core.tools import log
from ...integrations.models.pd.integration import SecretField


def get_token_limits():
    vault_client = VaultClient()
    secrets = vault_client.get_all_secrets()
    return json.loads(secrets.get('open_ai_azure_token_limits', ''))


class CapabilitiesModel(BaseModel):
    completion: bool = False
    chat_completion: bool = True
    embeddings: bool = False


class AIModel(BaseModel):
    id: str
    name: Optional[str]
    capabilities: CapabilitiesModel = CapabilitiesModel()
    token_limit: Optional[int]

    @validator('name', always=True, check_fields=False)
    def name_validator(cls, value, values):
        return values.get('model', value)

    @validator('token_limit', always=True, check_fields=False)
    def token_limit_validator(cls, value, values):
        if value:
            return value
        token_limits = get_token_limits()
        return token_limits.get(values.get('id'), 8096)


class IntegrationModel(BaseModel):
    api_token: SecretField | str
    model_name: str = 'gpt-35-turbo'
    models: List[AIModel] = []
    api_version: str = '2023-03-15-preview'
    api_base: str = "https://ai-proxy.lab.epam.com"
    api_type: str = "azure"
    temperature: float = 0
    max_tokens: int = 512
    top_p: float = 0.8
    stream: bool = False

    @root_validator(pre=True)
    def prepare_model_list(cls, values):
        models = values.get('models')
        if models and isinstance(models[0], str):
            values['models'] = [AIModel(id=model, name=model).dict(by_alias=True) for model in models]
        return values

    @property
    def token_limit(self):
        return next((model.token_limit for model in self.models if model.id == self.model_name), 8096)

    def get_token_limit(self, model_name):
        return next((model.token_limit for model in self.models if model.id == model_name), 8096)

    def check_connection(self, project_id=None):
        if not project_id:
            project_id = session_project.get()
        #
        settings = self.dict()
        #
        module = context.module_manager.module.open_ai_azure
        if module.ad_token_provider is None:
            settings["api_token"] = self.api_token.unsecret(project_id)
        else:
            settings["azure_ad_token"] = module.ad_token_provider()
        #
        return worker_client.ai_check_settings(
            integration_name=this.module_name,
            settings=settings,
        )

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
    temperature: float = 0.3
    max_tokens: int = 2048
    top_p: float = 0.8
