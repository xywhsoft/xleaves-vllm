# Ornith 1.5 双 V100 专项版本

## 已验证配置

两张 V100 PCIe 32GB，没有 NVLink；上下文上限为每请求 240,128 tokens。
最终部署使用下面第一种配置，第二种配置是独立验证过的长上下文方案。

| 配置 | 活跃请求上限 | 已验证完整长上下文驻留 | 同时全新长输入测试 |
| --- | ---: | ---: | ---: |
| high160 | 160 | 16 | 16 |
| full26 | 26 | 26 | 16 |

160 个活跃请求不意味着 160 路都能同时用满 240K。
输入和输出共同受单请求总上下文上限约束。长上下文已缓存结果与全新输入结果必须分开看。

准确的原始命令、环境变量和逐文件摘要保存在：

- [high160](evidence/workspace/results/qualified-adaptive416-0-final.json)
- [full26](evidence/workspace/results/qualified-adaptive416-1-final.json)
- [最终性能报告](evidence/workspace/reports/ornith-final535.html)
- [性能数据](evidence/workspace/reports/ornith-final535.json)
- [运维说明](evidence/workspace/reports/ornith-final535-operations.txt)

配置中的绝对路径来自原测试服务器。迁移时必须更新模型路径、Python 环境、
`PYTHONPATH`、原生扩展库路径和多模态配置路径，不能直接把原命令当作跨机器启动命令。
凭据由运行时传入。

## 复现环境与来源

原环境使用 Linux、Python 3.12、PyTorch 2.10 cu128、CUDA 12.8 和 CUTLASS 4.4.2。
使用 `uv` 管理环境，依赖快照中的原始 editable 路径也需要改为当前 checkout。

- [原始复现说明](evidence/REPRODUCTION.txt)
- [原始依赖快照](evidence/runtime-requirements.txt)
- [原始源码环境](evidence/source-environment.json)
- [上游基准上的专项补丁](evidence/onecat-serving.patch)
- [原生模块与第三方库来源](evidence/workspace/portability40/native-provenance.json)
- [FlashAttention 补丁](evidence/workspace/portability40/flash-attention-sm70.patch)

补丁对应 Git 已跟踪文件的变化；34 个当时未跟踪的新增源码文件已经直接放入本仓库。
当前代码已导入补丁结果，不需要再对本仓库重复应用 `onecat-serving.patch`。

额外的 FlashAttention 源码基准为
`zhinianqin/flash-attention-v100` 的 `c2eda5e6115b98c3ba4bfd181570668742eece22`。
该依赖的改动、已使用库摘要及运行时 Python 接口另有证据记录，不能只替换一个 `.so`
就宣称与已验证环境一致。

专项原生模块的构建入口在 [tools/build_ornith_sm70.py](../tools/build_ornith_sm70.py)。
示例命令必须在 Linux CUDA 环境内执行，且先准备带正确补丁的 FlashAttention 源码：

```bash
.venv/bin/python tools/build_ornith_sm70.py \
  --build-root ./ornith-native \
  --cutlass-root /path/to/cutlass-4.4.2 \
  --flash-attn-root /path/to/qualified-flash-attention
```

构建后更新配置中的原生库路径并重新运行数值、多模态、完整上下文和模型测试。
原有核心引擎与部分注意力二进制被复用过；归档源码与其摘要不构成完整干净构建验收。

## 完整证据包

`artifacts/onecat-final537-source-evidence-complete.tar.gz` 是独立完整验证包，
SHA-256 在上级 [README.xleaves.md](../README.xleaves.md) 和
[archive-provenance.json](archive-provenance.json) 中记录。

较大的 tokenizer、回放输入和原始测试数据只保留在完整包内，未放入普通 Git 跟踪文件。
包内包含它们的原始路径及 SHA-256。
可以解压到独立目录，再依据 `source-environment.json` 的路径映射恢复环境；
不要把整包直接覆盖在本仓库根目录。

历史测试脚本中的 PID、任务登记和机器准入记录属于原实验的证据。
在新机器上测试，应重新建立任务登记，不能复用这些历史 PID。
