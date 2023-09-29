from pylon.core.tools import log  # pylint: disable=E0611,E0401
from pylon.core.tools import web

from tools import rpc_tools
from ..models.integration_pd import IntegrationModel, AzureOpenAISettings
from ...integrations.models.pd.integration import SecretField
from pydantic import ValidationError


def _prepare_conversation(prompt_struct):
    conversation = []
    if prompt_struct.get('context'):
        conversation.append({
            "role": "system",
            "content": prompt_struct['context']
        })
    if prompt_struct.get('examples'):
        for example in prompt_struct['examples']:
            conversation.append({
                "role": "user",
                "content": example['input']
            })
            conversation.append({
                "role": "assistant",
                "content": example['output']
            })
    if prompt_struct.get('prompt'):
        conversation.append({
            "role": "user",
            "content": prompt_struct['prompt']
        })

    return conversation


def prepare_result(response):
    structured_result = {'messages': []}
    structured_result['messages'].append({
        'type': 'text',
        'content': response['choices'][0]['message']['content']
    })
    return structured_result


def prepare_stream_result(response):
    response = list(response)
    if not response:
        return {'type': "text", "content": ""}
     
    is_image = "diffusion" in response[0]['model'] 
    result = dict()
    if is_image:
        result['type'] = "image"
        result['content'] = []
    else:
        result['type'] = "text"
        result['content'] = ""
    for chunk in response:
        if is_image:
            attachments = chunk['choices'][0]['delta'].get('custom_content', {}).get('attachments', tuple())
            for attachment in attachments:
                if attachment.get('title') == 'image':
                    result['content'].append(attachment)
        else:
            result['content'] += chunk['choices'][0]['delta'].get('content', "")
    return result


class RPC:
    integration_name = 'open_ai_azure'

    @web.rpc(f'{integration_name}__predict')
    @rpc_tools.wrap_exceptions(RuntimeError)
    def predict(self, project_id, settings, prompt_struct):
        """ Predict function """
        import openai

        try:
            settings = IntegrationModel.parse_obj(settings)
        except ValidationError as e:
            return {"ok": False, "error": e}

        try:
            api_key = SecretField.parse_obj(settings.api_token).unsecret(project_id)
            openai.api_key = api_key
            openai.api_type = settings.api_type
            openai.api_base = settings.api_base
            openai.api_version = settings.api_version

            conversation = _prepare_conversation(prompt_struct)

            stream = settings.stream
            params = {
                'engine': settings.model_name,
                'temperature': settings.temperature,
                'top_p': settings.top_p,
                'messages': conversation,
            }
            if stream:
                params['stream'] = stream
            else:
                params['max_tokens'] = settings.max_tokens
            response = openai.ChatCompletion.create(**params)
            result = prepare_stream_result(response) if stream else prepare_result(response)

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
        import openai

        api_key = SecretField.parse_obj(payload['settings'].get('api_token', {})).unsecret(payload.get('project_id'))
        openai.api_key = api_key
        openai.api_type = payload['settings'].get('api_type')
        openai.api_base = payload['settings'].get('api_base')
        openai.api_version = payload['settings'].get('api_version')
        try:
            models = openai.Model.list()
        except Exception as e:
            log.error(str(e))
            models = []
        if models:
            models = models.get('data', [])
        return models
