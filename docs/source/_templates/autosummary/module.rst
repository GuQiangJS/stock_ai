{{ name | escape | underline}}

.. autodoc_member_order: 'alphabetical'

.. currentmodule:: {{fullname}}

{% block functions %}
{% if functions %}
.. rubric:: Methods
.. autosummary::
    :nosignatures:
    {% for item in functions %}
    {{ item }}
    {%- endfor %}
{% endif %}
{% endblock %}

{% if functions %}
.. rubric:: Methods
{% for item in functions %}
.. autofunction:: {{ item }}
{%- endfor %}
{% endif %}

{% block classes %}
{% if classes %}
.. rubric:: Class
.. autosummary::
    :toctree:
    :nosignatures:
    {% for item in classes %}
    {{ item }}
    {%- endfor %}
{% endif %}
{% endblock %}
