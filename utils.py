from .models.integration_pd import IntegrationModel

from pylon.core.tools import log
# from ..integrations.models.pd.integration import SecretField


def prepare_conversation(prompt_struct):
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


def prerare_text_prompt(prompt_struct):
    example_template = '\ninput: {input}\noutput: {output}'

    for example in prompt_struct['examples']:
        prompt_struct['context'] += example_template.format(**example)
    if prompt_struct['prompt']:
        prompt_struct['context'] += example_template.format(input=prompt_struct['prompt'], output='')

    return prompt_struct['context']


def _prepare_attachments(attachments):
    structured_attachments = []
    for attachment in attachments:
        if 'image' in attachment.get('type', ''):
            structured_attachments.append({
                'type': 'image',
                'content': attachment
            })
        if 'text' in attachment.get('type', '') or not attachment.get('type'):
            content = attachment['title'] + '\n\n' if attachment.get('title') else ''
            content += attachment['data'] if attachment.get('data') else ''
            content += '\n\n' + 'Reference URL: ' + attachment['reference_url'] if attachment.get('reference_url') else ''
            structured_attachments.append({
                'type': 'text',
                'content': content
            })
    return structured_attachments


def prepare_result(response):
    structured_result = {'messages': []}
    attachments = []

    if response['choices'][0]['message'].get('content'):
        structured_result['messages'].append({
            'type': 'text',
            'content': response['choices'][0]['message']['content']
        })
    else:
        attachments += response['choices'][0]['message'].get('custom_content', {}).get('attachments', [])
    attachments += response['choices'][0].get('custom_content', {}).get('attachments', [])

    if attachments:
        structured_result['messages'] += _prepare_attachments(attachments)

    return structured_result

def prepare_text_result(content):
    structured_result = {'messages': []}
    structured_result['messages'].append({
        'type': 'text',
        'content': content
    })
    return structured_result


def prepare_stream_result(response):
    structured_result = {'messages': []}
    text_content = ''
    attachments = []
    for chunk in response:
        if chunk['choices'][0]['delta'].get('content'):
            text_content += chunk['choices'][0]['delta']['content']
        else:
            attachments += chunk['choices'][0]['delta'].get('custom_content', {}).get('attachments', [])

    if text_content:
        structured_result['messages'].append({
            'type': 'text',
            'content': text_content
        })

    if attachments:
        structured_result['messages'] += _prepare_attachments(attachments)

    return structured_result


def predict_chat(project_id: int, settings: dict, prompt_struct: dict) -> str:
    import openai

    settings = IntegrationModel.parse_obj(settings)

    api_key = settings.api_token.unsecret(project_id)
    openai.api_key = api_key
    openai.api_type = settings.api_type
    openai.api_version = settings.api_version
    openai.api_base = settings.api_base

    conversation = prepare_conversation(prompt_struct)

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

    return prepare_stream_result(response) if stream else prepare_result(response)


def predict_text(project_id: int, settings: dict, prompt_struct: dict) -> str:
    import openai

    settings = IntegrationModel.parse_obj(settings)

    api_key = settings.api_token.unsecret(project_id)
    openai.api_key = api_key
    openai.api_type = settings.api_type
    openai.api_version = settings.api_version
    openai.api_base = settings.api_base

    text_prompt = prerare_text_prompt(prompt_struct)

    response = openai.Completion.create(
        engine=settings.model_name,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        top_p=settings.top_p,
        prompt=text_prompt
    )

    content = response['choices'][0]['text']
    log.info('completion_response %s', content)

    return prepare_text_result(content)
