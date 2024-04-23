#!/usr/bin/python3
# coding=utf-8

#   Copyright 2021 getcarrier.io
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

""" Module """
import json
from pylon.core.tools import log  # pylint: disable=E0611,E0401
from pylon.core.tools import module  # pylint: disable=E0611,E0401

from azure.identity import DefaultAzureCredential, get_bearer_token_provider  # pylint: disable=E0401

from tools import VaultClient  # pylint: disable=E0611,E0401

from .models.integration_pd import IntegrationModel


TOKEN_LIMITS = {
    'text-embedding-ada-002': None,
    'gpt-35-turbo': 4096,
    'gpt-35-turbo-16k': 16384,
    'gpt-4': 8192,
    'gpt-4-32k': 32768,
    'chat-bison@001': 8192,
    'ai21.j2-grande-instruct': 8191,
    'ai21.j2-jumbo-instruct': 8191,
    'anthropic.claude-instant-v1': 100000,
    'anthropic.claude-v1': 100000,
    'anthropic.claude-v2': 100000,
    'stability.stable-diffusion-xl': 77,
    'gpt-35-turbo-instruct': 4097
}


class Module(module.ModuleModel):
    """ Task module """

    def __init__(self, context, descriptor):
        self.context = context
        self.descriptor = descriptor
        #
        self.ad_token_provider = None

    def init(self):
        """ Init module """
        log.info('Initializing AI module')
        SECTION_NAME = 'ai'
        #
        self.descriptor.init_blueprint()
        self.descriptor.init_slots()
        self.descriptor.init_rpcs()
        self.descriptor.init_events()
        self.descriptor.init_api()
        #
        self.context.rpc_manager.call.integrations_register_section(
            name=SECTION_NAME,
            integration_description='Manage ai integrations',
        )
        self.context.rpc_manager.call.integrations_register(
            name=self.descriptor.name,
            section=SECTION_NAME,
            settings_model=IntegrationModel,
        )
        #
        vault_client = VaultClient()
        secrets = vault_client.get_all_secrets()
        if 'open_ai_azure_token_limits' not in secrets:
            secrets['open_ai_azure_token_limits'] = json.dumps(TOKEN_LIMITS)
            vault_client.set_secrets(secrets)
        #
        # Managed identity AD token
        #
        if "open_ai_azure_ad_token" in secrets:
            log.info("Using managed identity / AD for token")
            self.ad_token_provider = get_bearer_token_provider(
                DefaultAzureCredential(), secrets["open_ai_azure_ad_token"]
            )


    def deinit(self):
        """ De-init module """
        log.info('De-initializing')
