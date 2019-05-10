# stock_ai

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
* `pip install async_timeout` *(quantaxis需要，但是没有包含在安装请求中...)*
* `pip install pytest`
* `pip install coverage`