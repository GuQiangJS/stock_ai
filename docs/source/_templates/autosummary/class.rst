{{ name }}
{{ underline }}

.. autodoc_member_order: 'alphabetical'

.. currentmodule:: {{ module }}

.. autoclass:: {{name}}
    :members: __init__

{% block attributes %}
{% if attributes %}
.. rubric:: Attributes
.. autosummary::
    {% for item in all_attributes %}
    {%- if not item.startswith('_') %}
    {{ name }}.{{ item }}
    {%- endif -%}
    {%- endfor %}
{% endif %}
{% endblock %}

{% block methods %}
{% if methods %}
.. rubric:: Methods
.. autosummary::
    :nosignatures:
    {% for item in all_methods %}
    {%- if not item.startswith('_') or item in ['__call__'] %}
    {{ name }}.{{ item }}
    {%- endif -%}
    {%- endfor %}
{% endif %}
{% endblock %}

{% if attributes %}
.. rubric:: Attributes
{% for item in all_attributes %}
{%- if not item.startswith('_') %}
.. autoattribute:: {{ name }}.{{ item }}
{%- endif -%}
{%- endfor %}
{%- endif -%}

{% if methods %}
.. rubric:: Methods
{% for item in all_methods %}
{%- if not item.startswith('_') or item in ['__call__'] %}
.. automethod:: {{ name }}.{{ item }}
{%- endif -%}
{%- endfor %}
{%- endif -%}