# stock_ai

[![Build Status](https://travis-ci.com/GuQiangJS/stock_ai.svg?branch=master)](https://travis-ci.com/GuQiangJS/stock_ai)
[![Coverage Status](https://coveralls.io/repos/github/GuQiangJS/stock_ai/badge.svg?branch=master)](https://coveralls.io/github/GuQiangJS/stock_ai?branch=master)

---

## 环境相关

配置 `conda` 使用 proxy。
> `conda config --set proxy_servers.http http://192.168.9.180:1080`
> 
> `conda config --set proxy_servers.https https://192.168.9.180:1080`

* `conda create -n stock_ai python=3.7`
* `conda activate stock_ai`
* `pip install quantaxis` 或者 `pip --proxy=192.168.9.180:1080 install quantaxis`
* `pip install --upgrade tensorflow` 或者 `pip install --upgrade tensorflow-gpu`
* `pip install keras`
* `pip install -U scikit-learn`
* `pip install async_timeout` *(quantaxis需要，但是没有包含在安装请求中...)*
* `pip install pytest`
* `pip install coverage`

## 开发环境

### sphinxcontrib-napoleon

https://github.com/sphinx-contrib/napoleon

`pip install sphinxcontrib-napoleon`

### jupyter 插件

https://github.com/ipython-contrib/jupyter_contrib_nbextensions

```batchfile
conda install -c conda-forge jupyter_contrib_nbextensions
# Install nbextension files, and edits nbconvert config files
jupyter contrib nbextension install --user
# Install yapf for code prettify
pip install yapf
# Install autopep8
pip install autopep8
# Jupyter extensions configurator 
pip install jupyter_nbextensions_configurator
# Enable nbextensions_configurator
jupyter nbextensions_configurator enable --user
```

[Jupyter Notebook 小贴士](http://blog.leanote.com/post/carlking5019/Jupyter-Notebook-Tips)

> 如果选择了 `autopep8` ，还需要安装 `pip install autopep8`

查看插件是否启动 `http://localhost:8888/nbextensions`### pyecharts for mac 可能会遇到的问题

* 不显示图像。运行时浏览器后台有错误。![](images/QQ20190218-202434.png)

    解决方案：下载 [echarts.min.js](https://echarts.baidu.com/dist/echarts.min.js) 放至对应目录。

    > 例如：当前我本机的路径为:/usr/local/share/jupyter/nbextensions/,那么我就在这个文件夹下新增一个echarts的目录，将下载的js文件放进去。

### node.js

`conda install -c conda-forge nodejs`

### jupyter 支持进度条

https://tqdm.github.io/

`conda install -c conda-forge tqdm`

> 需要先安装 `ipywidgets`。`conda install -c conda-forge ipywidgets`
> 貌似暂时只能用在 jupyter notebook 中。

### pylint

https://www.pylint.org/#install

[如何使用Pylint 来规范Python 代码风格 - IBM](https://www.ibm.com/developerworks/cn/linux/l-cn-pylint/index.html)

`pip install pylint # see note`