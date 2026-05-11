# AI Agent 资料精讲：专业但通俗的学习讲义

更新时间：2026-05-11  
资料范围：OpenAI、Anthropic、LangGraph、MCP 官方资料与一手文档  
适合读者：已经知道“大模型能聊天”，但想真正理解 Agent 是怎么做事的人

---

## 0. 先给结论

Agent 不是“更会聊天的大模型”，而是：

> 让大模型在明确边界内，持续观察环境、选择工具、执行步骤、检查结果，并推进任务的工程系统。

如果把普通聊天机器人比作“顾问”，Agent 更像“带工具箱的助理”。  
顾问主要回答你；助理不只回答，还能查资料、调用函数、整理文件、写代码、运行测试、保存状态、遇到风险时停下来问你。

但这句话要补一句很重要的现实判断：

> Agent 的难点不在“调用模型”，而在“让它可靠、安全、可恢复地完成任务”。

> 参考原文：OpenAI Practical Guide: https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf；Anthropic: https://www.anthropic.com/engineering/building-effective-agents

---

## 1. Agent 到底是什么？

OpenAI 的实用指南把 Agent 描述为能代表用户完成任务的系统；普通 LLM 应用如果只是单次问答、分类、情绪分析，不算真正的 Agent。Anthropic 的文章也做了一个关键区分：Workflow 是固定代码路径，Agent 是由 LLM 动态决定过程和工具使用。

你可以这样理解：

| 类型 | 核心特点 | 例子 |
|---|---|---|
| 普通聊天机器人 | 用户问，模型答 | “解释一下 RAG 是什么” |
| Workflow | 路径提前写死 | “先总结，再翻译，再保存” |
| Agent | 模型根据任务和反馈决定下一步 | “帮我检查这个项目为什么启动失败，必要时看日志、改配置、跑测试” |

关键差别不是“有没有 LLM”，而是：

- 谁决定下一步？
- 能不能使用工具？
- 是否能根据工具结果继续调整？
- 是否有停止条件和安全边界？

一个简单但准确的 Agent 公式：

```text
Agent = Model + Instructions + Tools + Loop + State + Guardrails + Evaluation
```

> 参考原文：OpenAI Practical Guide: https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf；Anthropic: https://www.anthropic.com/engineering/building-effective-agents；OpenAI Agents SDK: https://openai.github.io/openai-agents-python/agents/

---

## 2. Agent 的核心部件

### 2.1 Model：大脑，但不是全部

Model 负责理解任务、推理、规划和决定是否调用工具。  
但是模型自己不会真的执行工具。它不会真的移动文件、发邮件、查数据库。它只能生成“我想调用这个工具，参数是这些”的请求。

所以 Agent 不是把所有事情都交给模型，而是把模型放进一个受控程序里。

> 参考原文：OpenAI Agents SDK - Agents: https://openai.github.io/openai-agents-python/agents/；OpenAI Practical Guide: https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf

### 2.2 Instructions：任务规则

Instructions 可以理解为 Agent 的工作守则。它应该说清楚：

- Agent 的角色是什么
- 目标是什么
- 可以做什么
- 不能做什么
- 什么时候必须停下来问用户
- 输出格式是什么
- 遇到错误怎么处理

坏指令像这样：

```text
你是一个智能助手，请帮用户完成任务。
```

好指令更像这样：

```text
你是文件整理 Agent。
你只能在用户指定目录内工作。
默认只生成整理计划，不移动文件。
不删除文件，不覆盖文件。
任何低置信度分类都放入人工审核列表。
移动前必须请求用户确认。
```

Instructions 不是装饰品。它是 Agent 行为的第一层约束。

> 参考原文：OpenAI Agents SDK - Agents: https://openai.github.io/openai-agents-python/agents/

### 2.3 Tools：Agent 的手

工具是 Agent 能接触外部世界的方式。

常见工具：

- 读文件
- 写文件
- 查询数据库
- 调用 HTTP API
- 搜索网页
- 执行命令
- 发送邮件
- 调用另一个 Agent

工具越强，风险越高。  
一个只能读文件的 Agent 风险很低；一个能删除文件、转账、发邮件的 Agent 必须有严格权限和人工确认。

工具定义要像给新人写函数文档一样清楚：

- 工具做什么
- 参数是什么
- 参数格式是什么
- 成功返回什么
- 失败返回什么
- 有哪些边界
- 什么时候不该调用

Anthropic 的文章特别强调，工具接口本身需要被认真设计。很多 Agent 的问题不是模型“不聪明”，而是工具描述模糊、参数设计容易出错。

> 参考原文：OpenAI Function Calling: https://developers.openai.com/api/docs/guides/function-calling；OpenAI Agents SDK - Tools: https://openai.github.io/openai-agents-python/tools/；Anthropic: https://www.anthropic.com/engineering/building-effective-agents

### 2.4 Loop：Agent 的发动机

Agent 最核心的结构是循环。

```text
收到用户目标
  -> 调用模型
  -> 模型决定：直接回答，还是调用工具
  -> 如果调用工具，程序执行工具
  -> 把工具结果交回模型
  -> 模型根据结果继续判断
  -> 直到完成、失败、超时、达到最大轮数，或需要人工介入
```

OpenAI Agents SDK 的 Runner 也是这个逻辑：调用 LLM，如果有最终输出就结束；如果有工具调用就执行工具并继续；如果有 handoff 就切换 Agent；如果超过最大轮数就报错。

伪代码：

```python
messages = [system_instructions, user_task]
turns = 0

while turns < max_turns:
    response = call_model(messages, tools=tools)

    if response.final_answer:
        return response.final_answer

    for tool_call in response.tool_calls:
        validate(tool_call)

        if is_high_risk(tool_call):
            ask_user_confirmation(tool_call)

        result = execute_tool(tool_call)
        messages.append(result)

    turns += 1

return "任务未完成：达到最大轮数，需要人工接管。"
```

这段循环就是 Agent 的骨架。框架只是帮你封装这段逻辑。

> 参考原文：OpenAI Agents SDK - Running agents: https://openai.github.io/openai-agents-python/running_agents/

### 2.5 State / Memory：记住当前任务和长期偏好

Agent 需要记住两类东西：

| 类型 | 作用 | 例子 |
|---|---|---|
| 短期状态 | 记住当前任务进度 | 已经读过哪些文件、当前执行到第几步 |
| 长期记忆 | 跨会话保存偏好或事实 | 用户喜欢中文回答、项目默认技术栈 |

LangGraph 的资料把 memory 分成 short-term memory 和 long-term memory。短期记忆通常跟一个线程或一次任务绑定；长期记忆跨会话存在。

但你要小心一个误区：

> 记忆不是把所有聊天记录无限塞给模型。

真正工程里要做的是：

- 摘要旧上下文
- 丢弃无关信息
- 保存结构化状态
- 把长期偏好放到可检索存储
- 对敏感记忆提供查看和删除能力

> 参考原文：LangGraph Memory: https://docs.langchain.com/oss/python/concepts/memory；LangGraph Persistence: https://docs.langchain.com/oss/python/langgraph/persistence

### 2.6 Guardrails：刹车和护栏

Guardrails 不是一句“请安全操作”。  
可靠的 Guardrails 应该落实在代码、权限和流程里。

常见 Guardrails：

- 输入检查：用户请求是否越权、是否恶意
- 输出检查：结果是否符合格式、是否包含敏感信息
- 工具检查：工具参数是否合法
- 权限检查：当前用户是否允许执行这个动作
- 风险分级：只读、可逆写入、不可逆写入、高价值操作
- 人工确认：高风险动作必须确认
- 最大轮数：防止 Agent 无限循环
- 日志和审计：能复盘每一步发生了什么

OpenAI 的 Agent 指南明确建议把 Guardrails 当成分层防御，而不是单点防御。

> 参考原文：OpenAI Agents SDK - Guardrails: https://openai.github.io/openai-agents-python/guardrails/；OpenAI Practical Guide: https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf

### 2.7 Evaluation：别靠感觉判断 Agent 好不好

Agent 的评估不能只看最终回答。你要看过程：

- 选了正确工具吗？
- 工具参数对吗？
- 该停的时候停了吗？
- 高风险动作有没有请求确认？
- 失败后有没有乱试？
- 有没有遵守边界？
- 输出能被程序稳定解析吗？

OpenAI 的 Agent Evals 文档建议从 traces 开始。Trace 可以记录一次运行中的模型调用、工具调用、handoff、guardrail 等过程，然后再用 grader 或数据集做可重复评估。

一句话：

> 没有评估的 Agent，只是“看起来能跑”；有评估的 Agent，才开始接近工程系统。

> 参考原文：OpenAI Agent Evals: https://developers.openai.com/api/docs/guides/agent-evals

---

## 3. Tool Calling：Agent 为什么能“动手做事”

Tool Calling 也叫 Function Calling。

最关键的一点：

> 模型不会执行函数，模型只提出函数调用请求；你的程序负责执行函数。

标准流程：

```text
1. 程序把工具列表发给模型
2. 模型判断需要哪个工具
3. 模型返回工具名和参数
4. 程序校验参数
5. 程序执行工具
6. 程序把工具结果发回模型
7. 模型基于结果继续回答或继续调用工具
```

例子：

用户问：

```text
帮我看看 D:\Downloads 里有哪些文件可以整理。
```

模型不应该凭空猜。它应该请求：

```json
{
  "tool": "list_files",
  "arguments": {
    "path": "D:/Downloads"
  }
}
```

你的程序执行 `list_files`，得到真实文件列表，再把结果交回模型。

然后模型才能说：

```json
{
  "summary": "发现 12 个图片、4 个 PDF、3 个压缩包。",
  "actions": [
    {
      "source": "D:/Downloads/a.png",
      "target": "D:/Downloads/Images/a.png",
      "confidence": "high"
    }
  ]
}
```

注意：模型给出的工具参数仍然必须校验。  
模型说要移动 `C:/Windows/System32`，程序必须拒绝。

> 参考原文：OpenAI Function Calling: https://developers.openai.com/api/docs/guides/function-calling；OpenAI Agents SDK - Tools: https://openai.github.io/openai-agents-python/tools/

---

## 4. Structured Outputs：让模型输出能被程序信任的数据

自然语言适合给人看，不适合给程序稳定处理。

如果你要让 Agent 输出计划、分类、风险等级、任务结果，最好用结构化输出。

比如不要只让模型说：

```text
我建议把图片放到 Images 文件夹。
```

而要让它输出：

```json
{
  "actions": [
    {
      "type": "move",
      "source": "D:/Downloads/photo.png",
      "target": "D:/Downloads/Images/photo.png",
      "risk": "low",
      "confidence": "high"
    }
  ],
  "requires_confirmation": true
}
```

OpenAI 的 Structured Outputs 资料强调，结构化输出可以让模型结果符合你定义的 JSON Schema。工程上这很重要，因为程序可以校验：

- 字段是否存在
- 类型是否正确
- 枚举值是否有效
- 是否缺少必要信息

一句话：

> 给人看的结果可以是自然语言；给程序继续执行的结果必须尽量结构化。

> 参考原文：OpenAI Structured Outputs: https://developers.openai.com/api/docs/guides/structured-outputs；OpenAI Agents SDK - Agents: https://openai.github.io/openai-agents-python/agents/

---

## 5. RAG：让 Agent 查资料，而不是瞎背

RAG 是 Retrieval-Augmented Generation，意思是“检索增强生成”。

它的目的不是让模型“记住所有资料”，而是在回答前先查资料。

基本流程：

```text
用户问题
  -> 检索相关文档片段
  -> 把片段放进上下文
  -> 模型基于片段回答
  -> 给出来源
```

OpenAI 的 File Search 和 Retrieval 文档里，核心概念是 vector store。你把文件上传到向量库，系统会把文件切块、向量化、建立索引。之后模型可以通过 file_search 工具查找相关内容。

你可以把 RAG 理解成：

> 给 Agent 配了一个资料柜。它回答前先去资料柜翻资料，而不是只靠脑子里的旧知识。

RAG 适合：

- 项目文档问答
- 公司知识库问答
- 法规、制度、流程查询
- 代码库说明文档
- PDF / DOCX / Markdown 笔记问答

RAG 不适合被神化。它仍然会出错：

- 检索结果可能不相关
- 文档可能过期
- 模型可能误读片段
- 没检索到时模型可能想补答案

所以 RAG Agent 必须有规则：

```text
如果资料中找不到答案，就明确说找不到。
不要编造来源。
回答必须引用对应片段或文件名。
```

> 参考原文：OpenAI Retrieval: https://platform.openai.com/docs/guides/retrieval；OpenAI File Search: https://platform.openai.com/docs/guides/tools-file-search/

---

## 6. MCP：让工具和上下文变成标准接口

MCP 是 Model Context Protocol。你可以先把它理解为：

> 给 AI 应用连接外部工具和上下文的一套标准协议。

MCP 里有几个角色：

| 概念 | 通俗解释 |
|---|---|
| Host | 使用 LLM 的应用，比如 IDE、聊天应用、Agent 客户端 |
| Client | Host 里的连接器 |
| Server | 提供工具、资源、提示词的服务 |
| Resources | 可读上下文，比如文件、数据库记录、文档 |
| Prompts | 可复用提示词模板 |
| Tools | 可被模型调用的函数能力 |

MCP 的价值在于标准化。  
以前每个 Agent 框架都要自己接数据库、浏览器、GitHub、文件系统。MCP 希望这些能力能像插件一样被统一暴露。

但 MCP 也强调安全：

- 用户要知道暴露了哪些工具
- 工具调用应该有清晰提示
- 高风险操作应该让用户确认
- 用户数据不能随便传给外部服务

所以 MCP 不是“让 Agent 放飞自我”，而是“用标准方式接工具，同时保留用户控制权”。

> 参考原文：MCP Specification: https://modelcontextprotocol.io/specification/2024-11-05/index；MCP Tools: https://modelcontextprotocol.io/specification/draft/server/tools

---

## 7. Memory：Agent 该记住什么，不该记住什么

Agent 的记忆要克制。

### 7.1 短期记忆

短期记忆负责当前任务。

例子：

- 用户刚刚要求整理哪个目录
- 已经列出了哪些文件
- 已经生成了什么计划
- 用户是否确认执行

短期记忆通常存在当前线程、会话状态或 checkpoint 里。

### 7.2 长期记忆

长期记忆跨会话存在。

例子：

- 用户偏好中文解释
- 用户喜欢先计划再执行
- 某个项目默认技术栈是 Next.js + Supabase
- 某个 Agent 默认不能删除文件

长期记忆应该可控，不应该偷偷无限保存。

### 7.3 三类常见长期记忆

| 类型 | 含义 | Agent 例子 |
|---|---|---|
| Semantic memory | 事实 | 用户的默认开发环境是 Windows |
| Episodic memory | 经验 | 上次某个任务失败，因为端口被占用 |
| Procedural memory | 做事规则 | 遇到危险文件操作时先给计划，不直接执行 |

记忆设计的核心问题不是“能不能记”，而是：

- 这条信息未来真的有用吗？
- 是否涉及隐私？
- 用户能不能查看和删除？
- 是否会过期？
- 是否会误导当前任务？

> 参考原文：LangGraph Memory: https://docs.langchain.com/oss/python/concepts/memory；LangGraph Persistence: https://docs.langchain.com/oss/python/langgraph/persistence

---

## 8. Guardrails：靠谱 Agent 必须有边界

Agent 越能干，越需要边界。

可以按风险给工具分级：

| 风险级别 | 工具例子 | 默认策略 |
|---|---|---|
| 低风险 | 读取目录、搜索文档、查看日志 | 可自动执行 |
| 中风险 | 创建文件、移动文件、修改草稿 | 先预览，必要时确认 |
| 高风险 | 删除文件、发邮件、支付、改权限 | 必须人工确认 |
| 禁止 | 泄露密钥、绕过权限、删除系统文件 | 直接拒绝 |

文件整理 Agent 的 Guardrails 示例：

```text
默认 dry-run。
不删除文件。
不覆盖已有文件。
不处理系统目录。
不处理隐藏文件。
每次最多处理 100 个文件。
低置信度文件必须进入人工审核列表。
移动文件前必须展示完整计划并等待确认。
```

这就是“工程里的安全感”：不是相信模型永远正确，而是假设它可能出错，然后把错误挡在边界内。

> 参考原文：OpenAI Agents SDK - Guardrails: https://openai.github.io/openai-agents-python/guardrails/；OpenAI Practical Guide: https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf

---

## 9. Multi-Agent：不要一上来就多 Agent

多 Agent 很诱人，但初学阶段不要急。

OpenAI 和 Anthropic 的资料都倾向一个原则：

> 先用简单方案，只有复杂度真的需要时再拆分。

什么时候考虑多 Agent？

- 单个 Agent 的 instructions 太长，条件分支太多
- 工具太多且相互相似，模型经常选错
- 任务天然需要不同专家角色
- 某些任务需要完全交给另一个 Agent 接管

常见模式：

| 模式 | 含义 | 适合场景 |
|---|---|---|
| Manager / agents as tools | 主 Agent 调用子 Agent，但主 Agent 保持控制权 | 写报告、代码审查、搜索汇总 |
| Handoffs | 一个 Agent 把任务交给另一个 Agent，由后者接管 | 客服分流、销售/退款/技术支持 |
| Orchestrator-workers | 主 Agent 动态拆任务，分配给多个 worker | 大型代码修改、多资料研究 |
| Evaluator-optimizer | 一个生成，一个评价，然后迭代 | 翻译、写作、搜索质量改进 |

初学建议：

```text
第一个月只做单 Agent。
等你能解释清楚单 Agent 的工具调用、状态、错误处理和评估，再碰多 Agent。
```

> 参考原文：Anthropic: https://www.anthropic.com/engineering/building-effective-agents；OpenAI Agents SDK - Agent orchestration: https://openai.github.io/openai-agents-python/multi_agent/；OpenAI Agents SDK - Handoffs: https://openai.github.io/openai-agents-python/handoffs/

---

## 10. 什么时候该用 Agent，什么时候不该用？

适合用 Agent：

- 任务步骤不固定
- 需要根据中间结果调整策略
- 需要调用多个工具
- 需要处理不完整信息
- 有明确成功标准
- 有可验证反馈，比如测试、日志、查询结果

不适合一上来用 Agent：

- 单次 LLM 调用就能解决
- 固定流程就能解决
- 对延迟和成本极度敏感
- 错误代价很高但没有人工确认
- 没有评估手段
- 工具权限太大但没有安全边界

一句工程判断：

> 如果规则流程能稳定解决，就先用规则流程；如果任务需要动态判断和工具反馈，再用 Agent。

> 参考原文：OpenAI Practical Guide: https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf；Anthropic: https://www.anthropic.com/engineering/building-effective-agents

---

## 11. 推荐学习路线：从项目中理解 Agent

### 阶段 1：普通 LLM 调用

目标：知道模型输入输出是什么。

你要做：

- 写一个命令行聊天程序
- 理解 system / user / assistant 消息
- 理解模型不会自己执行动作

产出：

```text
chat.py
```

### 阶段 2：结构化输出

目标：让模型输出能被程序解析的数据。

你要做：

- 定义 JSON Schema 或 Pydantic 模型
- 让模型输出任务计划
- 校验字段、类型、枚举

产出：

```text
plan_schema.py
```

### 阶段 3：第一个只读工具

目标：让模型基于真实环境做判断。

你要做：

- 写 `list_files(path)`
- 读取文件名、扩展名、大小、修改时间
- 把结果交给模型生成整理建议

产出：

```text
tools/filesystem.py
```

### 阶段 4：Agent Loop

目标：真正进入 Agent。

你要做：

- 模型决定是否调用工具
- 程序执行工具
- 工具结果回传模型
- 支持多轮
- 设置最大轮数

产出：

```text
agent_loop.py
```

### 阶段 5：本地文件整理 Agent

目标：做一个低风险但完整的小 Agent。

功能：

- 输入目录
- 列出文件
- 生成整理计划
- 标记低置信度项
- 默认 dry-run
- 用户确认后移动
- 不删除、不覆盖
- 写日志

这是你的第一个真正项目。

### 阶段 6：RAG 文档问答 Agent

目标：让 Agent 会查你的资料。

你要做：

- 准备 Markdown / PDF / DOCX 文档
- 切块、建立向量索引
- 用户提问时检索相关片段
- 回答时附来源

产出：

```text
doc_qa_agent/
```

### 阶段 7：Memory 和 Persistence

目标：让 Agent 可暂停、可恢复、有状态。

你要做：

- 保存当前任务状态
- 保存用户偏好
- 支持继续上次任务
- 区分短期记忆和长期记忆

### 阶段 8：Guardrails 和 Evals

目标：从“能跑”变成“可验证地更靠谱”。

你要做：

- 为工具加风险等级
- 高风险动作必须确认
- 准备 20 个测试任务
- 记录每次工具调用
- 统计成功率、误调用率、需要人工介入次数

### 阶段 9：再学框架

这时你再看：

- OpenAI Agents SDK
- LangGraph
- MCP
- CrewAI / AutoGen 等多 Agent 框架

你会清楚它们到底帮你封装了什么，而不是被名词牵着走。

> 参考原文：OpenAI Agents SDK: https://openai.github.io/openai-agents-python/；LangGraph Memory: https://docs.langchain.com/oss/python/concepts/memory；LangGraph Persistence: https://docs.langchain.com/oss/python/langgraph/persistence；MCP Specification: https://modelcontextprotocol.io/specification/2024-11-05/index

---

## 12. 第一个项目：文件整理 Agent 设计稿

### 12.1 项目目标

用户输入：

```text
帮我整理这个测试目录：D:\agent-test-files
```

Agent 输出：

```text
我发现 35 个文件。
建议移动 22 个，保留 8 个，人工确认 5 个。
默认不会执行移动，只展示计划。
```

### 12.2 工具列表

```text
list_files(path)
preview_plan(actions)
move_file(source, target)
write_log(event)
```

### 12.3 安全规则

```text
只允许操作用户指定目录。
禁止删除文件。
禁止覆盖已有文件。
禁止移动系统文件。
移动前必须展示 source 和 target。
用户输入 yes 之前不执行。
```

### 12.4 计划结构

```json
{
  "summary": "建议按图片、文档、压缩包分类。",
  "actions": [
    {
      "type": "move",
      "source": "D:/agent-test-files/a.png",
      "target": "D:/agent-test-files/Images/a.png",
      "reason": "图片文件",
      "confidence": "high",
      "risk": "medium"
    }
  ],
  "needs_review": [
    {
      "source": "D:/agent-test-files/unknown.bin",
      "reason": "扩展名无法判断",
      "suggestion": "保留原位"
    }
  ]
}
```

### 12.5 验收标准

第一版做到这些就合格：

- 能读取目录
- 能生成结构化计划
- 默认 dry-run
- 不删除文件
- 不覆盖文件
- 低置信度项进入人工确认
- 用户确认后才移动
- 有最大轮数
- 工具异常能停止
- 有日志

> 参考原文：OpenAI Practical Guide: https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf；OpenAI Function Calling: https://developers.openai.com/api/docs/guides/function-calling；OpenAI Agents SDK - Guardrails: https://openai.github.io/openai-agents-python/guardrails/

---

## 13. 你要建立的工程直觉

### 直觉 1：Agent 不是越自主越好

自主性越高，越需要边界。  
新手做 Agent，第一目标不是“让它什么都能做”，而是“让它只在允许范围内做对事”。

### 直觉 2：工具比 Prompt 更重要

Prompt 可以引导行为，但工具决定能力边界。  
如果工具设计糟糕，模型再强也容易乱用。

### 直觉 3：自然语言适合沟通，结构化数据适合执行

给用户解释可以自然语言。  
给程序执行必须 JSON / Schema / Typed Object。

### 直觉 4：失败路径要提前设计

Agent 必须知道：

- 工具失败怎么办
- 参数不合法怎么办
- 达到最大轮数怎么办
- 用户不确认怎么办
- 任务不清楚怎么办

没有失败路径的 Agent，不适合处理真实任务。

### 直觉 5：评估是核心能力

如果你不能证明 Agent 这版比上一版更好，那你只是在凭感觉调 Prompt。

> 参考原文：OpenAI Agent Evals: https://developers.openai.com/api/docs/guides/agent-evals；Anthropic: https://www.anthropic.com/engineering/building-effective-agents；OpenAI Practical Guide: https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf

---

## 14. 术语速查

| 术语 | 通俗解释 |
|---|---|
| Agent | 能围绕目标多步执行、调用工具、根据反馈调整的系统 |
| Workflow | 固定流程，路径由代码预先写好 |
| Tool Calling | 模型请求调用工具，程序负责执行 |
| Structured Outputs | 让模型按 JSON Schema 输出 |
| RAG | 回答前先检索外部资料 |
| Vector Store | 存放文档向量索引的资料库 |
| Memory | 保存任务状态或长期偏好 |
| Guardrails | 输入、输出、工具和权限的安全边界 |
| Handoff | 一个 Agent 把任务交给另一个 Agent |
| Trace | 一次 Agent 运行的完整过程记录 |
| Eval | 用测试集和评分规则评估 Agent |
| MCP | 连接 LLM 应用和外部工具/上下文的标准协议 |

> 参考原文：OpenAI Practical Guide: https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf；OpenAI Function Calling: https://developers.openai.com/api/docs/guides/function-calling；OpenAI Structured Outputs: https://developers.openai.com/api/docs/guides/structured-outputs；MCP Specification: https://modelcontextprotocol.io/specification/2024-11-05/index

---

## 15. 推荐阅读顺序

1. Anthropic：先理解 workflows 和 agents 的区别  
   https://www.anthropic.com/engineering/building-effective-agents

2. OpenAI Practical Guide：理解 Agent 的工程组成  
   https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf

3. OpenAI Function Calling：理解工具调用流程  
   https://developers.openai.com/api/docs/guides/function-calling

4. OpenAI Structured Outputs：理解结构化输出  
   https://developers.openai.com/api/docs/guides/structured-outputs

5. OpenAI Agents SDK：理解框架如何封装 Agent Loop  
   https://openai.github.io/openai-agents-python/agents/

6. OpenAI Running Agents：理解 Runner 循环和最大轮数  
   https://openai.github.io/openai-agents-python/running_agents/

7. OpenAI Guardrails：理解输入、输出和工具护栏  
   https://openai.github.io/openai-agents-python/guardrails/

8. OpenAI Retrieval / File Search：理解 RAG 和向量库  
   https://platform.openai.com/docs/guides/retrieval  
   https://platform.openai.com/docs/guides/tools-file-search/

9. LangGraph Memory：理解短期记忆和长期记忆  
   https://docs.langchain.com/oss/python/concepts/memory

10. LangGraph Persistence：理解 checkpoint、恢复和 human-in-the-loop  
    https://docs.langchain.com/oss/python/langgraph/persistence

11. MCP Specification：理解工具和上下文的标准接口  
    https://modelcontextprotocol.io/specification/2024-11-05/index

12. OpenAI Agent Evals：理解如何评估 Agent  
    https://developers.openai.com/api/docs/guides/agent-evals

---

## 16. 最后给你的学习判断

你现在最该做的不是继续看十几个框架，而是：

```text
手写一个最小 Agent。
只给它一个只读工具。
让它生成结构化计划。
再慢慢加入执行、确认、日志、记忆和评估。
```

真正理解 Agent 的标志不是你能说出多少名词，而是你能回答这些问题：

- 模型什么时候会调用工具？
- 工具参数怎么校验？
- 工具失败后怎么处理？
- 高风险动作怎么拦住？
- 状态保存在哪里？
- 如何恢复中断任务？
- 如何证明这版 Agent 比上一版更可靠？

把这些问题跑通，你就不是在“听 Agent 的概念”，而是在真正学 Agent。

> 参考原文：OpenAI Practical Guide: https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf；OpenAI Agents SDK: https://openai.github.io/openai-agents-python/；Anthropic: https://www.anthropic.com/engineering/building-effective-agents
