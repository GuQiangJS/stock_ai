* `conf.py`文件中增加以下配置：
	```
	extensions = [
	'sphinx.ext.autodoc',
	'sphinx.ext.autosummary',
	]
	autosummary_generate = True
	autosummary_imported_members = True
	```
	
* `srouce\_templates\autosummary` 文件夹中增加 `module.rst` 文件。内容如下：
	```
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
	```
	* `autodoc_member_order`表示按照名称排序。
	* 自动将所有公开方法加入 `autosummary`，并且以简单方式显示方法定义（不显示参数）

* `index.rst`文件中写入如下配置：
	```
	.. autosummary::
	   :toctree: api/

	   stock_ai.data_processor
	```
	意思是在 调用 `make html` 时会自动创建 `source\api\stock_ai.data_processor.rst` 文件。
	
使用以上三步以后，会先自动创建对应的rst文件。然后才会根据rst文件创建html。不再需要一个一个手动创建rst。