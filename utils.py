# FIXME: ChatCompletion and Completion are not adapted for managed identity and openai > 1.0.0

from collections import deque
from openai import ChatCompletion, Completion
import tiktoken
from .models.integration_pd import IntegrationModel
from .models.request_body import ChatCompletionRequestBody, CompletionRequestBody

from pylon.core.tools import log


def init_openai(settings, project_id):
    return {
        'api_key': settings.api_token.unsecret(project_id),
        'api_type': settings.api_type,
        'api_version': settings.api_version,
        'api_base': settings.api_base
    }


def num_tokens_from_messages(messages: list, model: str):
    """Return the number of tokens used by a list of messages.
    See: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        log.warning("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        log.warning("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        log.warning("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        tokens_per_message = 4
        tokens_per_name = -1
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    # num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def limit_conversation(
        conversation: dict, model_name: str, max_response_tokens: int, token_limit: int
        ) -> list:
        limited_conversation = []
        remaining_tokens = token_limit - max_response_tokens
        remaining_tokens -= 3  # every reply is primed with <|start|>assistant<|message|>

        context_tokens = num_tokens_from_messages(conversation['context'], model_name)
        remaining_tokens -= context_tokens

        if remaining_tokens < 0:
            raise Exception(
'There are no enough tokens to form messages for ChatCompletion. \
Try using a lower value for the token limit parameter.'
)

        limited_conversation.extend(conversation['context'])

        input_tokens = num_tokens_from_messages(conversation['input'], model_name)
        remaining_tokens -= input_tokens
        if remaining_tokens < 0:
            return limited_conversation

        final_examples = []
        for example in conversation['examples']:
            example_tokens = num_tokens_from_messages([example], model_name)
            remaining_tokens -= example_tokens
            if remaining_tokens < 0:
                if len(final_examples) % 2:
                    final_examples.pop()  # remove incomplete example if present
                return limited_conversation + final_examples + conversation['input']
            final_examples.append(example)

        limited_conversation.extend(final_examples)

        final_history = deque()
        for message in reversed(conversation['chat_history']):
            message_tokens = num_tokens_from_messages([message], model_name)
            remaining_tokens -= message_tokens
            if remaining_tokens < 0:
                return limited_conversation + list(final_history) + conversation['input']
            final_history.appendleft(message)
        limited_conversation.extend(final_history)

        limited_conversation.extend(conversation['input'])
        return limited_conversation


def prepare_conversation(
        prompt_struct: dict, model_name: str, max_response_tokens: int, token_limit: int,
        check_limits: bool = True
        ) -> list:
    conversation = {
        'context': [],
        'examples': [],
        'chat_history': [],
        'input': []
    }

    if prompt_struct.get('context'):
        conversation['context'].append({
            "role": "system",
            "content": prompt_struct['context']
        })
    if prompt_struct.get('examples'):
        for example in prompt_struct['examples']:
            conversation['examples'].append({
                "role": "user",
                "name": "example_user",
                "content": example['input']
            })
            if example.get("output", None):
                conversation['examples'].append({
                    "role": "assistant",
                    "name": "example_assistant",
                    "content": example['output']
                })
    if prompt_struct.get('chat_history'):
        for message in prompt_struct['chat_history']:
            conversation['chat_history'].append({
                "role": "user" if message['role'] == 'user' else "assistant",
                "content": message['content']
            })
    if prompt_struct.get('prompt'):
        conversation['input'].append({
            "role": "user",
            "content": prompt_struct['prompt']
        })

    if check_limits:
        return limit_conversation(conversation, model_name, max_response_tokens, token_limit)

    return conversation['context'] + conversation['examples'] + conversation['chat_history'] + conversation['input']



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


def limit_messages(messages: list, model_name: str, max_response_tokens: int, token_limit: int) -> list:
    conversation = {
        'context': [],
        'examples': [],
        'chat_history': [],
        'input': []
    }
    for idx, message in enumerate(messages):
        if message['role'] == 'system' and not message.get('name'):
            conversation['context'].append(message)
        if message.get("name") in ("example_user", "example_assistant"):
            conversation['examples'].append(message)
        if message['role'] == 'user' and idx != len(messages) - 1:
            conversation['chat_history'].append(message)
        if message['role'] == 'assistant':
            conversation['chat_history'].append(message)
    if messages[-1]['role'] == 'user':
        conversation['input'].append(messages[-1])

    return limit_conversation(conversation, model_name, max_response_tokens, token_limit)


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
    settings = IntegrationModel.parse_obj(settings)
    init_settings = init_openai(settings, project_id)

    stream = settings.stream
    token_limit = settings.token_limit
    max_tokens = settings.max_tokens if not stream else 0

    conversation = prepare_conversation(
        prompt_struct, settings.model_name, max_tokens, token_limit)

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
    response = ChatCompletion.create(**params, **init_settings)

    return prepare_stream_result(response) if stream else prepare_result(response)


def predict_chat_from_request(project_id: int, settings: dict, request_data: dict) -> str:
    params = ChatCompletionRequestBody.validate(request_data).dict(exclude_unset=True)
    settings = IntegrationModel.parse_obj(settings)
    init_settings = init_openai(settings, project_id)

    token_limit = settings.get_token_limit(params['deployment_id'])
    max_tokens = params.get('max_tokens', 0)
    if params.get('messages'):
        params['messages'] = limit_messages(
            params['messages'], params['deployment_id'], max_tokens, token_limit
            )

    return ChatCompletion.create(**params, **init_settings)


def predict_from_request(project_id: int, settings: dict, request_data: dict) -> str:
    params = CompletionRequestBody.validate(request_data).dict(exclude_unset=True)
    settings = IntegrationModel.parse_obj(settings)
    init_settings = init_openai(settings, project_id)
    return Completion.create(**params, **init_settings)


def predict_text(project_id: int, settings: dict, prompt_struct: dict) -> str:
    settings = IntegrationModel.parse_obj(settings)
    init_settings = init_openai(settings, project_id)

    text_prompt = prerare_text_prompt(prompt_struct)

    response = Completion.create(
        engine=settings.model_name,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        top_p=settings.top_p,
        prompt=text_prompt,
        **init_settings
    )

    content = response['choices'][0]['text']
    log.info('completion_response %s', content)

    return prepare_text_result(content)
