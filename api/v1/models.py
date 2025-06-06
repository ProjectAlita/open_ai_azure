from flask import request

try:
    from pydantic.v1 import ValidationError
except:  # pylint: disable=W0702
    from pydantic import ValidationError

from tools import api_tools

from ...models.integration_pd import IntegrationModel


class ProjectAPI(api_tools.APIModeHandler):
    ...

class AdminAPI(api_tools.APIModeHandler):
    ...


class API(api_tools.APIBase):
    url_params = [
        '<string:mode>/<int:project_id>',
        '<string:mode>/<string:project_id>',
        '<int:project_id>',
        '<string:project_id>',
    ]

    mode_handlers = {
        'default': ProjectAPI,
        'administration': AdminAPI,
    }

    def post(self, project_id, **kwargs):
        try:
            project_id = int(project_id)
        except:  # pylint: disable=W0702
            project_id = None

        try:
            settings = IntegrationModel.parse_obj(request.json)
        except ValidationError as e:
            return e.errors(), 400

        check_connection_response = settings.check_connection()
        if check_connection_response is not True:
            return [{'loc': ['check_connection'], 'msg': check_connection_response}], 400

        models = settings.refresh_models(project_id)
        return models, 200
