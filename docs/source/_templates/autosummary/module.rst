{{ fullname | escape | underline}}

.. autodoc_member_order: 'alphabetical'

.. currentmodule:: {{fullname}}

{% block functions %}
{% if functions %}
.. autosummary::
    :nosignatures:
    {% for item in functions %}
    {{ item }}
    {%- endfor %}
{% endif %}
{% endblock %}

{% if functions %}
{% for item in functions %}
.. autofunction:: {{ item }}
{%- endfor %}
{% endif %}

{% block classes %}
{% if classes %}
.. autosummary::
    :toctree:
    :nosignatures:
    {% for item in classes %}
    {{ item }}
    {%- endfor %}
{% endif %}
{% endblock %}
