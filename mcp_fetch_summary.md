# MCP Fetch 网页内容抓取服务器 - 总结

## 概述

**MCP Fetch** 是一个模型上下文协议（Model Context Protocol）服务器，专门用于网页内容抓取。该服务器使大型语言模型能够从网页中检索和处理内容，并将 HTML 转换为 Markdown 格式，以便更容易地使用和处理。

- **项目名称**: Fetch MCP Server
- **中文名称**: Fetch网页内容抓取
- **许可证**: MIT License
- **GitHub 仓库**: https://github.com/modelcontextprotocol/servers/tree/main/src/fetch
- **调用量**: 221,711,835 次
- **星标数**: 534
- **查看次数**: 315,893 次

## 核心功能

### 主要工具

**`fetch`** - 从互联网抓取 URL 并将其内容提取为 Markdown 格式

**参数说明**:
- `url` (字符串, 必需): 要抓取的 URL
- `max_length` (整数, 可选): 返回的最大字符数，默认值为 5000
- `start_index` (整数, 可选): 从此字符索引开始提取内容，默认值为 0
- `raw` (布尔值, 可选): 获取未经 Markdown 转换的原始 HTML 内容，默认值为 false

### 特性

1. **分块读取**: 通过 `start_index` 参数，模型可以分块读取网页内容，直到找到所需的信息
2. **内容截断**: fetch 工具会截断响应，但可以通过参数控制返回的内容长度
3. **格式转换**: 自动将 HTML 转换为 Markdown 格式，便于处理
4. **原始内容**: 可选择获取原始 HTML 内容

## 安装方法

### 方法一：使用 uv（推荐）

使用 [`uv`](https://docs.astral.sh/uv/) 时不需要特定的安装步骤，可以直接使用 [`uvx`](https://docs.astral.sh/uv/guides/tools/) 运行：

```bash
uvx mcp-server-fetch
```

### 方法二：使用 PIP

通过 pip 安装：

```bash
pip install mcp-server-fetch
```

安装后运行：

```bash
python -m mcp_server_fetch
```

### 可选：安装 Node.js

安装 Node.js 后，fetch 服务器会使用更健壮的 HTML 简化器。

## 配置

### Claude.app 配置

在 Claude 设置中添加以下配置：

#### 使用 uvx

```json
"mcpServers": {
  "fetch": {
    "command": "uvx",
    "args": ["mcp-server-fetch"]
  }
}
```

#### 使用 Docker

```json
"mcpServers": {
  "fetch": {
    "command": "docker",
    "args": ["run", "-i", "--rm", "mcp/fetch"]
  }
}
```

#### 使用 pip 安装

```json
"mcpServers": {
  "fetch": {
    "command": "python",
    "args": ["-m", "mcp_server_fetch"]
  }
}
```

### 自定义配置

#### robots.txt 处理

- **默认行为**: 如果请求来自模型（通过工具），服务器会遵守网站的 robots.txt 文件；如果请求由用户发起（通过提示），则不会遵守
- **禁用**: 在配置的 `args` 列表中添加 `--ignore-robots-txt` 参数可以禁用此行为

#### 用户代理（User-Agent）

**默认用户代理**:
- 模型请求: `ModelContextProtocol/1.0 (Autonomous; +https://github.com/modelcontextprotocol/servers)`
- 用户请求: `ModelContextProtocol/1.0 (User-Specified; +https://github.com/modelcontextprotocol/servers)`

**自定义**: 在配置的 `args` 列表中添加 `--user-agent=YourUserAgent` 参数

## 调试

使用 MCP 检查器来调试服务器：

### uvx 安装

```bash
npx @modelcontextprotocol/inspector uvx mcp-server-fetch
```

### 本地开发

```bash
cd path/to/servers/src/fetch
npx @modelcontextprotocol/inspector uv run mcp-server-fetch
```

## 使用场景

1. **网页内容提取**: 从网页中提取文本内容并转换为 Markdown
2. **信息检索**: 帮助 LLM 获取最新的网页信息
3. **内容分析**: 分块读取大型网页，逐步分析内容
4. **数据抓取**: 为 AI 模型提供网页访问能力

## 贡献

项目鼓励贡献，包括：
- 添加新工具
- 增强现有功能
- 改进文档

更多 MCP 服务器示例和实现模式请参考：
https://github.com/modelcontextprotocol/servers

## 许可证

mcp-server-fetch 采用 MIT 许可证，允许自由使用、修改和分发。

## 总结

MCP Fetch 是一个强大的网页内容抓取工具，专为大型语言模型设计。它提供了简单易用的接口，能够将网页 HTML 转换为 Markdown 格式，并支持分块读取和自定义配置。通过多种安装方式（uv、pip、Docker）和灵活的配置选项，可以轻松集成到各种 AI 应用中。

---

**来源**: https://modelscope.cn/mcp/servers/@modelcontextprotocol/fetch  
**抓取时间**: 2025年1月
