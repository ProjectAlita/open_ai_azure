from pylon.core.tools import log  # pylint: disable=E0611,E0401
from pylon.core.tools import web
from traceback import format_exc

from tools import rpc_tools, worker_client, this, context, SecretString
from ..models.integration_pd import AIModel, AzureOpenAISettings
from ..utils import predict_chat, predict_text, predict_chat_from_request, predict_from_request

try:
    from pydantic.v1 import ValidationError
except:  # pylint: disable=W0702
    from pydantic import ValidationError


class RPC:
    integration_name = 'open_ai_azure'

    @web.rpc(f'{integration_name}__predict')
    @rpc_tools.wrap_exceptions(RuntimeError)
    def predict(self, project_id, settings, prompt_struct):
        """ Predict function """
        models = settings.get('models', [])
        capabilities = next((model['capabilities'] for model in models if model['id'] == settings['model_name']), {})

        try:
            if capabilities.get('chat_completion'):
                log.info('Using chat prediction for model: %s', settings['model_name'])
                result = predict_chat(project_id, settings, prompt_struct)
            elif capabilities.get('completion'):
                log.info('Using completion(text) prediction for model: %s', settings['model_name'])
                result = predict_text(project_id, settings, prompt_struct)
            else:
                raise Exception(f"Model {settings['model_name']} does not support chat or text completion")
        except Exception as e:
            log.error(format_exc())
            return {"ok": False, "error": f"{type(e)}: {str(e)}"}

        return {"ok": True, "response": result}


    @web.rpc(f'{integration_name}__chat_completion')
    @rpc_tools.wrap_exceptions(RuntimeError)
    def chat_completion(self, project_id, settings, request_data):
        """ Chat completion function """
        try:
            result = predict_chat_from_request(project_id, settings, request_data)
        except Exception as e:
            log.error(str(e))
            return {"ok": False, "error": f"{str(e)}"}

        return {"ok": True, "response": result}

    @web.rpc(f'{integration_name}__completion')
    @rpc_tools.wrap_exceptions(RuntimeError)
    def completion(self, project_id, settings, request_data):
        """ Completion function """
        try:
            result = predict_from_request(project_id, settings, request_data)
        except Exception as e:
            log.error(str(e))
            return {"ok": False, "error": f"{str(e)}"}

        return {"ok": True, "response": result}

    @web.rpc(f'{integration_name}__parse_settings')
    @rpc_tools.wrap_exceptions(RuntimeError)
    def parse_settings(self, settings) -> dict:
        try:
            settings = AzureOpenAISettings.parse_obj(settings)
        except ValidationError as e:
            return {"ok": False, "error": e}
        return {"ok": True, "item": settings}


    @web.rpc(f'{integration_name}_set_models', 'set_models')
    @rpc_tools.wrap_exceptions(RuntimeError)
    def set_models(self, payload: dict):
        settings = {
            "api_base": payload["settings"]["api_base"],
            "api_version": payload["settings"]["api_version"],
        }
        #
        module = context.module_manager.module.open_ai_azure
        if module.ad_token_provider is None:
            if isinstance(payload['settings'].get('api_token', {}), SecretString):
                token_field = payload['settings'].get('api_token')
            else:
                token_field = SecretString(
                    payload['settings'].get('api_token', {})
                )
            #
            settings["api_token"] = token_field.unsecret(payload.get('project_id'))
        else:
            settings["azure_ad_token"] = module.ad_token_provider()
        #
        raw_models = worker_client.ai_get_models(
            integration_name=this.module_name,
            settings=settings,
        )
        #
        return [AIModel(**model).dict() for model in raw_models]
