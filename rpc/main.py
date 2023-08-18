from pylon.core.tools import log  # pylint: disable=E0611,E0401
from pylon.core.tools import web

from tools import rpc_tools
from ..models.integration_pd import IntegrationModel
from pydantic import ValidationError
import openai
from ...integrations.models.pd.integration import SecretField


class RPC:
    integration_name = 'open_ai_azure'

    @web.rpc(f'{integration_name}__predict')
    @rpc_tools.wrap_exceptions(RuntimeError)
    def predict(self, project_id, settings, text_prompt):
        """ Prepare dusty config for this scanner """
        try:
            settings = IntegrationModel.parse_obj(settings)
        except ValidationError as e:
            return {"ok": False, "error": e}

        try:
            api_key = SecretField.parse_obj(settings.api_token).unsecret(project_id)
            openai.api_key = api_key
            openai.api_type = "azure"
            openai.api_base = settings.api_base
            openai.api_version = settings.api_version

            response = openai.ChatCompletion.create(
                engine=settings.model_name,
                temperature=settings.temperature,
                messages=[
                    {
                        "role": "assistant",
                        "content": text_prompt,
                    }
                ]
            )
            result = response['choices'][0]['message']['content']    
        except Exception as e:
            log.error(str(e))
            return {"ok": False, "error": "Error in connection"}
    
        return {"ok": True, "response": result}

