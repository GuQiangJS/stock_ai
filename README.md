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
* `pip install async_timeout` *(quantaxis需要，但是没有包含在安装请求中...)*
* `pip install pytest`
* `pip install coverage`