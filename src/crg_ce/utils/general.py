from typing import cast

from jinja2 import StrictUndefined, Template

from crg_ce.resources import read_resource


def resolve_template(template_path_or_content: str) -> Template:
    """Resolves a template file name or template string content to a Jinja template with StrictUndefined checking

    Args:
        template_path_or_content (str): path to the template under src/crg_ce

    Raises:
        e: FileNotFoundError if the argument appears to be a path but the file does not exist

    Returns:
        Template: a Jinja template from file
    """
    try:
        template_content: str = read_resource(template_path_or_content)
        return cast(Template, Template(template_content, undefined=StrictUndefined))
    except FileNotFoundError as e:
        if template_path_or_content.endswith(".j2") or template_path_or_content.endswith(".txt"):
            raise e
        return cast(Template, Template(template_path_or_content, undefined=StrictUndefined))
