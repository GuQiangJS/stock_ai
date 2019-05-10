1. 设置 docstring 为 google风格。
    > File | Settings | Tools | Python Integrated Tools for Windows and Linux
    > 
    > PyCharm | Preferences | Tools | Python Integrated Tools for macOS
    > 
    > `Ctrl+Alt+S`

2. 安装 [yapf](https://github.com/google/yapf)
    > `pip install yapf`
    
3. 配置 yapf
    新增一个 External Tools。 配置如下：
    * **Program:** C:\Users\GuQiang\Anaconda3\envs\stock_ai\Scripts\yapf.exe
    * **Parameters:** -i --style=google $FilePath$
    * **Working directory:** $ProjectFileDir$
    
4. Keymap | External Tools 中找到刚刚新建的操作。为其分配 shortcut。
*我自己分配的是Ctrl+Alt+L，替换原本的Reformat Code*。