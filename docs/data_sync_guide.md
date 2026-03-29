# `processed` 数据同步说明

## 1. 结论

`processed/` 目前约 `2.86 GB`，共 `2989` 个 `.nii.gz` 文件。

这批数据**不建议直接提交到普通 Git 仓库**，原因是：

1. 数据总体量较大，会明显拖慢仓库拉取和克隆；
2. 组员后续每次 `git clone` 都会被迫下载整批 MRI；
3. GitHub 普通仓库不适合作为这种体量数据的长期分发方式。

推荐方案：

1. **代码、文档、脚本**：走 GitHub 仓库同步；
2. **`processed` 数据**：单独压缩后走网盘、共享盘或校园云盘同步；
3. 组员收到数据后，直接解压到项目根目录下，保证目录结构为 `ML_SVM/processed/...`。

## 2. 推荐的组内同步方式

最稳妥的方式是：

1. 由一名成员在本机把 `processed/` 打包成压缩包；
2. 上传到 OneDrive、Google Drive、123 云盘、百度网盘或学校共享盘；
3. 将下载链接发给组员；
4. 组员下载后解压到项目根目录；
5. 用校验脚本确认文件数量与总大小一致。

## 3. 已提供的脚本

本项目提供两个 PowerShell 脚本：

1. `scripts/package_processed.ps1`
   用于打包 `processed/` 并生成 SHA256 校验值。
2. `scripts/verify_processed.ps1`
   用于组员在本地验证 `processed/` 是否完整。

## 4. 打包命令

在项目根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\package_processed.ps1
```

默认会生成：

1. `processed_dataset.zip`
2. `processed_dataset.zip.sha256.txt`
3. `processed_manifest.csv`

## 5. 组员验证命令

组员拿到数据并解压后，在项目根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\verify_processed.ps1
```

默认会检查：

1. 是否存在 `processed/`
2. 文件数是否为 `2989`
3. 总大小是否约为 `2860991576` 字节

## 6. 是否要用 Git LFS

理论上可以用 Git LFS，但当前并不推荐直接这么做，原因是：

1. `2.86 GB` 数据对 GitHub LFS 配额很敏感；
2. 后续若继续增加数据，LFS 成本会更高；
3. 组内课程项目通常没有必要让所有 MRI 永久挂在 Git 历史中。

因此当前最实用的策略仍然是：

**Git 管代码，网盘管数据。**
