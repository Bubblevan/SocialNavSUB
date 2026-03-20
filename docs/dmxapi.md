OpenAI Responses API 请求格式介绍
? 简介
OpenAI 在 2025 年 3 月 11 日推出了全新的 Responses API，定位为 Chat Completions 的继任者。该 API 旨在简化开发流程并增强多模态能力，主要特性包括：

? 支持文本和图像输入，以及文本输出
? 创建与模型的有状态交互，将先前响应的输出用作输入
? 通过文件搜索、网络搜索、计算机使用等内置工具扩展模型的能力
? 使用函数调用允许模型访问外部系统和数据
? 为什么推出 /v1/responses 接口？
1. 功能更全面
在 /v1/chat/completions（仅支持对话生成）基础上，新增：

结构化输出
内建工具调用（如网页搜索、文档检索、代码执行、模拟操作等）
形成统一的 "Agent" 接口
2. 简化开发模式
无需手动编写 glue code（解析模型输出 → 执行工具 → 反馈）
模型可自主调用工具，服务器端处理 agent 循环逻辑
3. 逐步替换 Assistants API
OpenAI 宣布将在 2026 年上半年淘汰 Assistants API
鼓励开发者迁移到 Responses API
4. 官方 SDK 推广主打
最新 OpenAI SDK 已优先推荐 client.responses.create(...)
Chat Completions 接口保留兼容性
? /v1/chat/completions 对比 /v1/responses
对比项	/v1/chat/completions	/v1/responses
推出时间	行业标准接口，早在 2023 年已广泛使用	2025 年 3 月正式发布
状态管理	? 无状态，每次请求需带全对话消息历史	? 支持 store: true，通过 previous_response_id 维护上下文
工具集成	仅通过 function calling，需开发者管理工具调用流程	内建工具能力（web_search、file_search、MCP 工具）直接集成
多模态支持	可处理文本、function calls，图像需额外接口	支持文本、图像生成流、工具调用、人机交互/MCP 等一体化
流式能力	支持文本流式输出	支持文本、图像流式输出
工具扩展机制	通过 functions 参数自定义调用	可接入第三方 MCP 服务器，声明后模型可动态调用和审批
兼容性	会继续长期支持	不兼容 Assistants API（后者将于 2026 年上半年退役）
推荐使用场景	适合简单 chat 与 function calling 应用	构建带工具交互、Agent 或长期会话全托管应用
? /v1/responses 接口发展时间线
时间	重要事件
2025 年 3 月 11 日	OpenAI 正式发布 Responses API，宣布替代 Assistants API，并提供 web search、file search、computer-use 等工具
2025 年 3 月中旬	开发者论坛及社区（如 Simon Willison 博客）推出详细对比和说明文章
2025 年 5 月 21 日	OpenAI 宣布进一步扩展 Responses API 工具集（图像生成、代码解释、MCP 支持等）
2025 年 7 月 10 日	Azure OpenAI 版本作为 Preview 上线 Responses API
