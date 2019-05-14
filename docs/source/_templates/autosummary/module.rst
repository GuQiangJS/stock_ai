{{ fullname | escape | underline}}

.. autodoc_member_order: 'alphabetical'

.. currentmodule:: {{fullname}}

.. autosummary::
    :nosignatures:
    {% for item in functions %}
    {{ item }}
    {%- endfor %}

.. automodule:: {{fullname}}
    :members:
