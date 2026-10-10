# xLeaves Ornith / V100 专项优化存档

本仓库保存 Ornith 1.5 在双 Tesla V100 PCIe 32GB 上的最终验证源码，基于
[1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)。
专项优化原样导入，未重新格式化已验证的实现。

- 上游基准提交：`0dc468217feda5a5a50dfc87e4d9525a1daf5574`。
- 基准提交及其之前的 **20,031 个提交**保留原始 SHA、作者、时间和父提交关系。
- `origin`：`https://gitee.com/xywhsoft/xleaves-vllm`。
- `upstream`：`https://github.com/1CatAI/1Cat-vLLM.git`。
- `master` 保存专项优化；`upstream/main` 保存下载时的上游主线。
- 上游标签已保留；专项存档标签为 `ornith-v100-qualified-20261009`。
- 存档日期：2026-10-10；最终部署验证日期：2026-10-09。

上游原始说明保留在 [README.md](README.md)。
专项复现说明见 [ornith-optimization/README.md](ornith-optimization/README.md)。
逐文件来源与 SHA-256 见
[archive-provenance.json](ornith-optimization/archive-provenance.json)。

## 存档内容

`vllm/`、`csrc/`、`tools/` 等目录包含完整上游代码和 **49 个专项变更源码文件**。
这些文件逐字节匹配最终部署版本的验证包，包括多模态 GGUF 加载、SM70 MoE
路由和输出累加、量化 KV 注意力 CUDA 内核、prefill 缓冲复用、调度保护和采样优化。

`ornith-optimization/evidence/` 保存依赖快照、原始补丁、启动配置、性能报告、
原生扩展来源、测试和复现脚本。历史候选实验脚本保存在证据目录中，不代表已启用。

`ornith-optimization/artifacts/` 保存完整的 235,390,296 字节源码与证据压缩包，
包括原有 Linux 二进制、完整 tokenizer、大型回放输入和测试原始数据。
此目录已被 Git 忽略，普通 `git push` 不会上传这个大文件。

完整包 SHA-256：

```text
949ae6958b31328ce6ff4209a08a95df0f099ea779205d731ad0d0c0c21e65d4
```

模型权重和运行时凭据不在本仓库内。

## 历史与 fork 信息

服务器原始克隆是浅克隆，只有一个提交；本次从上游补齐了基准版本的完整祖先历史。
专项优化原本是未提交的工作区修改，本次以独立存档提交接在上游基准之后。
各轮实验没有既有的独立 Git 提交，因此不会伪造实验提交或修改上游提交时间。

Git 的提交历史可以随正常推送保留。GitHub/Gitee 页面上的 fork 网络关系属于平台
元数据，添加 `upstream` 或推送同一批 Git 对象不会自动建立该关系。
原平台的推送者、推送时间、活动流、审计日志和 PR 讨论也不随 Git 提交迁移。
本仓库通过上游 remote、基准提交、许可证和来源记录保留可追溯的派生信息。

本次仅完成本地存档，未推送远端。后续明确需要上传时，可以普通推送当前分支和
专项标签，无需强制推送，也不应对已有远端执行 `--mirror`。

```bash
git push -u origin master
git push origin ornith-v100-qualified-20261009
```

若还要发布上游标签，可以另行推送选定标签。独立于 `master` 的其他上游分支和
平台操作日志不在上述两条命令的上传范围内。

## 验证范围

本次存档检查了压缩包 SHA-256、49 个源码文件的 SHA-256、102 个 Python
源码文件的语法，以及 Git 基准和非浅历史。没有在 Windows 上重新进行 GPU 测试。
原有模型与算子验证结果见证据目录；完整引擎此前沿用部分已构建二进制，
尚未完成整个引擎从干净环境编译后的重新验收。
