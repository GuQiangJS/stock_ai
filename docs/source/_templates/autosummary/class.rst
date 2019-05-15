{{ fullname }}
{{ underline }}

.. autodoc_member_order: 'alphabetical'

.. currentmodule:: {{ module }}

.. autoclass:: {{name}}
    :members: __init__

Attributes
----------
{% block attributes %}
{% if attributes %}
.. autosummary::
    {% for item in all_attributes %}
    {%- if not item.startswith('_') %}
    {{ name }}.{{ item }}
    {%- endif -%}
    {%- endfor %}
{% endif %}
{% endblock %}

Methods
-------
{% block methods %}
{% if methods %}
.. autosummary::
    :nosignatures:
    {% for item in all_methods %}
    {%- if not item.startswith('_') or item in ['__call__'] %}
    {{ name }}.{{ item }}
    {%- endif -%}
    {%- endfor %}
{% endif %}
{% endblock %}

{% for item in all_attributes %}
{%- if not item.startswith('_') %}
.. autoattribute:: {{ name }}.{{ item }}
{%- endif -%}
{%- endfor %}

{% for item in all_methods %}
{%- if not item.startswith('_') or item in ['__call__'] %}
.. automethod:: {{ name }}.{{ item }}
{%- endif -%}
{%- endfor %}