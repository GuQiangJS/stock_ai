创建`.travis.yml`文件。示例在最后。

*如果需要自动部署文档，则需要安装 `doctr`*。`pip install doctr`。
安装后，在命令行中执行`doctr configure`。根据提示生成*deploy key*。

如果需要在travis环境中跳过pytest的单元测试。可以使用以下attribute标记。

`@pytest.mark.skipif("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true", reason="Skipping this test on Travis CI.")`

> **reason一定要写**。

示例：

```yml
language: python

python:
  - "3.6"

# This gives Doctr the key we've generated
env:
  global:
    secure: "<your secure key from Doctr here>"

matrix:
  include:
  - python: '3.6'
    env: DOCBUILD=true #如果是python3.6版本的情况下，构建文档

install:
  - pip install --upgrade pylint pytest pytest-pylint pytest-runner pytest-cov #测试相关
  - pip install pyecharts==0.5.11 #QUANTAXIS1.4版本相关的pyecharts
  - pip install pyecharts-snapshot==0.1.10 #QUANTAXIS1.4版本相关的pyecharts-snapshot
  - pip install quantaxis==1.4.0 #坑爹的QUANTAXIS，1.4版本以上强制使用mongodb存储设置
  - pip install --upgrade tensorflow
  - pip install coveralls #统计代码覆盖率
  - if [[ $DOCBUILD ]]; then
  # 如果是确定构建文档的标记下
    pip install sphinx sphinx_rtd_theme doctr sphinxcontrib-napoleon;#生成文档用
  fi
script:
  - pytest --version
  - pytest --cov=stock_ai test/ #使用pytest测试代码
  - |
  if [[ $DOCBUILD ]]; then
    cd docs
    make html
    cd ..
    doctr deploy . --build-tags
  fi
after_success:
  - coveralls #统计代码覆盖率

```