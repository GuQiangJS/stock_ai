# stock_ai

---

## 环境相关

配置 `conda` 使用 proxy。
> `conda config --set proxy_servers.http http://192.168.9.180:1080`
> 
> `conda config --set proxy_servers.https https://192.168.9.180:1080`

1. `conda create -n stock_ai python=3.7`
2. `conda activate stock_ai`
3. `pip install quantaxis` 或者 `pip --proxy=192.168.9.180:1080 install quantaxis`
4. `pip install --upgrade tensorflow` 或者 `pip install --upgrade tensorflow-gpu`